"""Unit tests for AdaptiveWrongWayModule and SpatialTrafficFlowField."""

import pytest
import numpy as np

from road_surveillance.events.adaptive_wrong_way import (
    SpatialTrafficFlowField,
    AdaptiveWrongWayModule,
    FlowDirectionModel,
)
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics


def test_traffic_flow_field_learning_and_persistence():
    flow = SpatialTrafficFlowField(grid_rows=2, grid_cols=2, min_samples_to_persist=4, min_coherence_to_persist=0.70)

    # 4 observations of vehicles moving downward (0.0, 1.0) in top-left cell (norm_x=0.2, norm_y=0.2)
    for f in range(4):
        flow.observe_motion(0.2, 0.2, (0.0, 1.0), frame_idx=f)

    model = flow.get_flow_model_for_position(0.2, 0.2)
    assert model is not None
    assert model.is_persisted is True
    assert model.direction_vector[0] == pytest.approx(0.0, abs=1e-3)
    assert model.direction_vector[1] == pytest.approx(1.0, abs=1e-3)
    assert model.coherence >= 0.99

    # Traffic disappears! Model must remain active and persisted in memory
    flow.reset()
    persisted_model = flow.get_flow_model_for_position(0.2, 0.2)
    assert persisted_model is not None
    assert persisted_model.is_persisted is True
    assert persisted_model.direction_vector[1] == pytest.approx(1.0, abs=1e-3)


def test_adaptive_wrong_way_ignores_correct_flow():
    module = AdaptiveWrongWayModule(min_speed_kmh=12.0, min_track_age=4, min_consecutive_frames=5)
    frame_shape = (1080, 1920)

    # Establish flow moving East (1.0, 0.0) across 6 observations
    for f in range(6):
        module.flow_field.observe_motion(0.5, 0.5, (1.0, 0.0), frame_idx=f)

    # A car moving East (1.0, 0.0) at 50 km/h with track_age = 10
    tr = Track(track_id=1, class_id=2, class_label="car", track_age=10, current_bbox_xywh=(900, 500, 100, 60), detection_confidence=0.85)
    kin = TrackKinematics(
        track_id=1,
        frame_idx=10,
        pts_ns=333333330,
        ground_pos=(0.0, 0.0),
        velocity_vector=(13.88, 0.0),
        speed_mps=13.88,
        speed_kmh=50.0,
        acceleration_vector=(0.0, 0.0),
        acceleration_mag=0.0,
        heading_deg=0.0,
        heading_vector=(1.0, 0.0),
        stationary_duration_sec=0.0,
        lane_id=None,
        is_in_active_lane=True,
    )

    # Process 10 frames
    for f in range(10):
        cands = module.process_frame(
            frame_idx=10 + f,
            pts_ns=(10 + f) * 33333333,
            video_id="test_vid",
            tracks={1: tr},
            kinematics={1: kin},
            frame_shape=frame_shape,
        )
        assert len(cands) == 0  # No wrong-way triggered


def test_adaptive_wrong_way_triggers_on_persistent_opposition():
    module = AdaptiveWrongWayModule(min_speed_kmh=12.0, min_track_age=4, min_consecutive_frames=6)
    frame_shape = (1080, 1920)

    # Establish dominant traffic flow moving South (0.0, 1.0)
    for f in range(8):
        module.flow_field.observe_motion(0.5, 0.5, (0.0, 1.0), frame_idx=f)

    # Wrong-way car traveling North (0.0, -1.0) at 60 km/h
    tr_opp = Track(track_id=99, class_id=2, class_label="car", track_age=10, current_bbox_xywh=(950, 520, 110, 70), detection_confidence=0.90)
    kin_opp = TrackKinematics(
        track_id=99,
        frame_idx=20,
        pts_ns=666666660,
        ground_pos=(0.0, 0.0),
        velocity_vector=(0.0, -16.66),
        speed_mps=16.66,
        speed_kmh=60.0,
        acceleration_vector=(0.0, 0.0),
        acceleration_mag=0.0,
        heading_deg=180.0,
        heading_vector=(0.0, -1.0),  # Exact opposite of (0.0, 1.0) -> cos = -1.0
        stationary_duration_sec=0.0,
        lane_id=None,
        is_in_active_lane=True,
    )

    all_cands = []
    # Run 10 frames
    for f in range(10):
        cands = module.process_frame(
            frame_idx=20 + f,
            pts_ns=(20 + f) * 33333333,
            video_id="test_vid",
            tracks={99: tr_opp},
            kinematics={99: kin_opp},
            frame_shape=frame_shape,
        )
        all_cands.extend(cands)

    # After 6 consecutive frames of opposition, it must trigger
    assert len(all_cands) > 0
    trigger = all_cands[0]
    assert trigger.event_type in ("wrong_way_vehicle", "wrong_way_hazard")
    assert trigger.primary_track_id == 99
    assert trigger.signals["directional_cosine"] == pytest.approx(-1.0, abs=1e-3)
    assert trigger.confidence >= 0.85
    assert trigger.signals["consecutive_violation_frames"] >= 6
