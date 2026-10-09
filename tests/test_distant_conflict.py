"""Unit tests for Distant Trajectory Conflict Detection Logic (Iteration D)."""

import pytest
from road_surveillance.events.candidates.distant_conflict import DistantTrajectoryConflictDetector
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence


def test_distant_conflict_triggers_on_converging_trajectories():
    detector = DistantTrajectoryConflictDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=3,
        deceleration_threshold_ms2=-3.5,
        tca_max_sec=2.0,
        dca_metric_max_m=3.5,
        dist_metric_max_m=18.0,
        min_closing_speed_kmh=10.0,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.track_age = 5
    tr2.track_age = 5
    tr1.ground_history = [(0.0, 0.0)] * 5
    tr2.ground_history = [(10.0, 0.0)] * 5

    # Non-overlapping bounding boxes (separated by 10 meters)
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (300, 100, 50, 40)

    # Closing at 10 m/s relative, TCA=1.0s, DCA=0.5m, Track 2 decelerating at -5.0 m/s^2
    conv = TrajectoryConvergence(1, 2, distance_m=10.0, is_converging=True, tca_sec=1.0, dca_m=0.5)
    kin1 = TrackKinematics(1, 20, 200000000, (0, 0), (5, 0), 5.0, 18.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(2, 20, 200000000, (10, 0), (-5, 0), 5.0, 18.0, (-5, 0), -5.0, 180, (-1, 0), 0, None, True)

    cands = detector.evaluate(20, 200000000, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.event_type == "distant_trajectory_conflict"
    assert cand.signals["trajectory_conflict"] is True
    assert set(cand.track_ids) == {1, 2}


def test_distant_conflict_diverging_paths_rejected():
    detector = DistantTrajectoryConflictDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=3,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.track_age = 5
    tr2.track_age = 5
    tr1.ground_history = [(0.0, 0.0)] * 5
    tr2.ground_history = [(10.0, 0.0)] * 5
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (300, 100, 50, 40)

    # Diverging: moving away from each other
    conv = TrajectoryConvergence(1, 2, distance_m=10.0, is_converging=False, tca_sec=float("inf"), dca_m=10.0)
    kin1 = TrackKinematics(1, 20, 200000000, (0, 0), (-5, 0), 5.0, 18.0, (0, 0), 0.0, 180, (-1, 0), 0, None, True)
    kin2 = TrackKinematics(2, 20, 200000000, (10, 0), (5, 0), 5.0, 18.0, (-5, 0), -5.0, 0, (1, 0), 0, None, True)

    cands = detector.evaluate(20, 200000000, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)
    assert len(cands) == 0


def test_distant_conflict_high_dca_miss_rejected():
    detector = DistantTrajectoryConflictDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=3,
        dca_metric_max_m=3.5,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.track_age = 5
    tr2.track_age = 5
    tr1.ground_history = [(0.0, 0.0)] * 5
    tr2.ground_history = [(10.0, 20.0)] * 5
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (300, 300, 50, 40)

    # Trajectories miss by 15 meters (DCA=15.0m > 3.5m)
    conv = TrajectoryConvergence(1, 2, distance_m=22.0, is_converging=True, tca_sec=1.0, dca_m=15.0)
    kin1 = TrackKinematics(1, 20, 200000000, (0, 0), (5, 0), 5.0, 18.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(2, 20, 200000000, (10, 20), (0, -5), 5.0, 18.0, (-5, 0), -5.0, 270, (0, -1), 0, None, True)

    cands = detector.evaluate(20, 200000000, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)
    assert len(cands) == 0


def test_distant_conflict_no_deceleration_rejected():
    detector = DistantTrajectoryConflictDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=3,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.track_age = 5
    tr2.track_age = 5
    tr1.ground_history = [(0.0, 0.0)] * 5
    tr2.ground_history = [(10.0, 0.0)] * 5
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (300, 100, 50, 40)

    # Cruising without deceleration
    conv = TrajectoryConvergence(1, 2, distance_m=10.0, is_converging=True, tca_sec=1.0, dca_m=0.5)
    kin1 = TrackKinematics(1, 20, 200000000, (0, 0), (5, 0), 5.0, 18.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(2, 20, 200000000, (10, 0), (-5, 0), 5.0, 18.0, (0, 0), 0.0, 180, (-1, 0), 0, None, True)

    cands = detector.evaluate(20, 200000000, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)
    assert len(cands) == 0
