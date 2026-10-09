"""Unit tests for Collision Candidate Detection Logic."""

import pytest
from road_surveillance.events.candidates.collision import CollisionCandidateDetector
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence


def test_collision_candidate_triggers_on_multi_signals():
    detector = CollisionCandidateDetector(
        distance_decay_threshold_m=1.0,
        tca_threshold_sec=2.0,
        deceleration_threshold_ms2=-3.0,
        post_proximity_distance_m=4.0,
    )
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (105, 105, 50, 40)  # Close overlap

    # Frame 0: distance = 10m
    conv0 = TrajectoryConvergence(1, 2, distance_m=10.0, is_converging=True, tca_sec=1.5, dca_m=1.0)
    kin1_0 = TrackKinematics(1, 0, 0, (0, 0), (5, 0), 5.0, 18.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2_0 = TrackKinematics(2, 0, 0, (10, 0), (-5, 0), 5.0, 18.0, (0, 0), 0.0, 180, (-1, 0), 0, None, True)

    detector.evaluate(0, 0, "v1", {1: tr1, 2: tr2}, {1: kin1_0, 2: kin2_0}, [conv0], lm)

    # Frame 1: distance rapidly collapsed to 2m (< 4.0m threshold), sudden deceleration anomaly (-6.0 m/s^2)
    conv1 = TrajectoryConvergence(1, 2, distance_m=2.0, is_converging=True, tca_sec=0.2, dca_m=0.5)
    kin1_1 = TrackKinematics(1, 1, int(1e8), (4, 0), (0, 0), 0.0, 0.0, (-6, 0), -6.0, 0, (1, 0), 0, None, True)
    kin2_1 = TrackKinematics(2, 1, int(1e8), (6, 0), (0, 0), 0.0, 0.0, (-6, 0), -6.0, 180, (-1, 0), 0, None, True)

    cands = detector.evaluate(1, int(1e8), "v1", {1: tr1, 2: tr2}, {1: kin1_1, 2: kin2_1}, [conv1], lm)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.event_type == "collision_candidate"
    assert cand.signals["decreasing_distance"] is True
    assert cand.signals["sudden_deceleration"] is True
    assert cand.signals["post_event_proximity"] is True
    assert set(cand.track_ids) == {1, 2}


def test_closure_corridor_rejected_on_open_space():
    """Verify F169 open-space conditions (d=15.75m, IoU=0.00) are rejected."""
    detector = CollisionCandidateDetector(is_camera_calibrated=True)
    lm = LaneManager()

    tr1 = Track(10, 2, "car")
    tr2 = Track(22, 2, "car")
    tr1.track_age = 10
    tr2.track_age = 10
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (300, 100, 50, 40)  # Open space, no IoU

    # Seed persistent closure
    pair_key = (10, 22)
    detector.closure_persist_counts[pair_key] = 3
    detector.last_closure_frame[pair_key] = 0

    conv = TrajectoryConvergence(10, 22, distance_m=15.75, is_converging=True, tca_sec=0.5, dca_m=1.0)
    kin1 = TrackKinematics(10, 1, int(1e8), (0, 0), (10, 0), 10.0, 36.0, (-15, 0), -15.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(22, 1, int(1e8), (15.75, 0), (0, 0), 0.0, 0.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)

    cands = detector.evaluate(1, int(1e8), "v1", {10: tr1, 22: tr2}, {10: kin1, 22: kin2}, [conv], lm)
    assert len(cands) == 0


def test_closure_corridor_rejected_on_positive_acceleration():
    """Verify candidate with positive acceleration (+60 m/s^2) is rejected."""
    detector = CollisionCandidateDetector(is_camera_calibrated=True)
    lm = LaneManager()

    tr1 = Track(10, 2, "car")
    tr2 = Track(22, 2, "car")
    tr1.track_age = 10
    tr2.track_age = 10
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (110, 100, 50, 40)  # Some bbox overlap

    pair_key = (10, 22)
    detector.closure_persist_counts[pair_key] = 3
    detector.last_closure_frame[pair_key] = 0

    # Distance 2.5m (requires closure corridor since standard max is 2.0m)
    conv = TrajectoryConvergence(10, 22, distance_m=2.5, is_converging=True, tca_sec=0.5, dca_m=1.0)
    # Positive acceleration (+60.0 m/s^2) - acceleration, not compressive deceleration!
    kin1 = TrackKinematics(10, 1, int(1e8), (0, 0), (15, 0), 15.0, 54.0, (60, 0), 60.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(22, 1, int(1e8), (2.5, 0), (10, 0), 10.0, 36.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)

    cands = detector.evaluate(1, int(1e8), "v1", {10: tr1, 22: tr2}, {10: kin1, 22: kin2}, [conv], lm)
    assert len(cands) == 0


def test_closure_corridor_admitted_on_terminal_physical_contact():
    """Verify candidate with IoU=0.311, a=-60 m/s^2 triggers collision_candidate via closure corridor."""
    detector = CollisionCandidateDetector(is_camera_calibrated=True)
    lm = LaneManager()

    tr1 = Track(7, 2, "car")
    tr2 = Track(14, 2, "car")
    tr1.track_age = 10
    tr2.track_age = 10

    # Frame 0: Approaching from 6.0m
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (150, 100, 50, 40)
    conv0 = TrajectoryConvergence(7, 14, distance_m=6.0, is_converging=True, tca_sec=0.5, dca_m=0.5)
    kin1_0 = TrackKinematics(7, 0, 0, (0, 0), (10, 0), 10.0, 36.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2_0 = TrackKinematics(14, 0, 0, (6.0, 0), (0, 0), 0.0, 0.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    detector.evaluate(0, 0, "v1", {7: tr1, 14: tr2}, {7: kin1_0, 14: kin2_0}, [conv0], lm)

    # Frame 1: Contact (IoU ~ 0.31), distance 2.5m, compressive deceleration (-60 m/s^2)
    tr2.current_bbox_xywh = (120, 100, 50, 40)
    pair_key = (7, 14)
    detector.closure_persist_counts[pair_key] = 3
    detector.last_closure_frame[pair_key] = 0

    conv1 = TrajectoryConvergence(7, 14, distance_m=2.5, is_converging=True, tca_sec=0.2, dca_m=0.5)
    kin1_1 = TrackKinematics(7, 1, int(1e8), (0, 0), (5, 0), 5.0, 18.0, (-60, 0), -60.0, 0, (1, 0), 0, None, True)
    kin2_1 = TrackKinematics(14, 1, int(1e8), (2.5, 0), (0, 0), 0.0, 0.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)

    cands = detector.evaluate(1, int(1e8), "v1", {7: tr1, 14: tr2}, {7: kin1_1, 14: kin2_1}, [conv1], lm)
    assert len(cands) == 1
    assert cands[0].signals["is_kinematic_closure_corridor"] is True
    assert set(cands[0].track_ids) == {7, 14}


def test_closure_corridor_uncalibrated_pixel_proximity():
    """Verify uncalibrated candidate with norm_px <= 0.50 and compressive deceleration triggers."""
    detector = CollisionCandidateDetector(is_camera_calibrated=False)
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.track_age = 10
    tr2.track_age = 10

    # Frame 0: Initial separation
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (140, 100, 50, 40)
    conv0 = TrajectoryConvergence(1, 2, distance_m=10.0, is_converging=True, tca_sec=0.5, dca_m=0.0)
    kin1_0 = TrackKinematics(1, 0, 0, (0, 0), (10, 0), 10.0, 36.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2_0 = TrackKinematics(2, 0, 0, (10, 0), (0, 0), 0.0, 0.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    detector.evaluate(0, 0, "v1", {1: tr1, 2: tr2}, {1: kin1_0, 2: kin2_0}, [conv0], lm)

    # Frame 1: Centroid distance 15 px with box height 40 px -> norm_pixel_dist = 0.375 <= 0.50
    tr2.current_bbox_xywh = (115, 100, 50, 40)
    pair_key = (1, 2)
    detector.closure_persist_counts[pair_key] = 3
    detector.last_closure_frame[pair_key] = 0

    conv1 = TrajectoryConvergence(1, 2, distance_m=2.0, is_converging=True, tca_sec=0.2, dca_m=0.0)
    kin1_1 = TrackKinematics(1, 1, int(1e8), (0, 0), (5, 0), 5.0, 18.0, (-8, 0), -8.0, 0, (1, 0), 0, None, True)
    kin2_1 = TrackKinematics(2, 1, int(1e8), (2, 0), (0, 0), 0.0, 0.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)

    cands = detector.evaluate(1, int(1e8), "v1", {1: tr1, 2: tr2}, {1: kin1_1, 2: kin2_1}, [conv1], lm)
    assert len(cands) == 1
    assert cands[0].signals["is_kinematic_closure_corridor"] is True
