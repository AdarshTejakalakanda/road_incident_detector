"""Unit tests for Wrong-Way Vehicle Detection Logic."""

import pytest
from road_surveillance.events.candidates.wrong_way import WrongWayDetector
from road_surveillance.calibration.lane_manager import Lane, LaneManager
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics


def test_wrong_way_triggers_on_persistent_opposition():
    # Northbound lane
    lane = Lane("L_N", "Northbound", True, (0.0, 1.0), [(-10, 0), (10, 0), (10, 50), (-10, 50)])
    lm = LaneManager([lane])

    detector = WrongWayDetector(min_speed_kmh=5.0, cos_threshold=-0.5, min_consecutive_frames=3)
    track = Track(track_id=1, class_id=2, class_label="car")

    # Moving South (0, -1) in Northbound lane at 30 km/h -> cos = -1.0
    kin = TrackKinematics(
        track_id=1,
        frame_idx=0,
        pts_ns=0,
        ground_pos=(0.0, 25.0),
        velocity_vector=(0.0, -8.33),
        speed_mps=8.33,
        speed_kmh=30.0,
        acceleration_vector=(0.0, 0.0),
        acceleration_mag=0.0,
        heading_deg=270.0,
        heading_vector=(0.0, -1.0),
        stationary_duration_sec=0.0,
        lane_id="L_N",
        is_in_active_lane=True,
    )

    # Frame 1: count=1 -> no trigger
    c1 = detector.evaluate(1, int(1e8), "v1", {1: track}, {1: kin}, [], lm)
    assert len(c1) == 0

    # Frame 2: count=2 -> no trigger
    c2 = detector.evaluate(2, int(2e8), "v1", {1: track}, {1: kin}, [], lm)
    assert len(c2) == 0

    # Frame 3: count=3 -> trigger!
    c3 = detector.evaluate(3, int(3e8), "v1", {1: track}, {1: kin}, [], lm)
    assert len(c3) == 1
    assert c3[0].event_type == "wrong_way_vehicle"
    assert c3[0].signals["directional_cosine"] == -1.0


def test_wrong_way_ignores_correct_direction():
    lane = Lane("L_N", "Northbound", True, (0.0, 1.0), [(-10, 0), (10, 0), (10, 50), (-10, 50)])
    lm = LaneManager([lane])
    detector = WrongWayDetector(min_speed_kmh=5.0, cos_threshold=-0.5, min_consecutive_frames=2)
    track = Track(track_id=1, class_id=2, class_label="car")

    # Moving North (0, 1) -> cos = 1.0
    kin = TrackKinematics(
        track_id=1,
        frame_idx=0,
        pts_ns=0,
        ground_pos=(0.0, 25.0),
        velocity_vector=(0.0, 8.33),
        speed_mps=8.33,
        speed_kmh=30.0,
        acceleration_vector=(0.0, 0.0),
        acceleration_mag=0.0,
        heading_deg=90.0,
        heading_vector=(0.0, 1.0),
        stationary_duration_sec=0.0,
        lane_id="L_N",
        is_in_active_lane=True,
    )

    for i in range(5):
        cands = detector.evaluate(i, int(i * 1e8), "v1", {1: track}, {1: kin}, [], lm)
        assert len(cands) == 0
