"""Unit tests for Single-Vehicle Anomaly Detection Logic (Iteration C)."""

import pytest
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics


def test_single_vehicle_severe_barrier_impact():
    detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=4,
        severe_impact_decel_ms2=-14.0,
    )
    lm = LaneManager()

    tr = Track(1, 2, "car")
    tr.track_age = 10
    tr.current_bbox_xywh = (200, 200, 80, 50)
    tr.ground_history = [(10.0, 10.0)] * 10

    # Warmup frames with normal speed
    for f in range(5):
        kin = TrackKinematics(1, f, f * 100000000, (10, 10), (10, 0), 10.0, 36.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
        cands = detector.evaluate(f, f * 100000000, "v1", {1: tr}, {1: kin}, [], lm)
        assert len(cands) == 0

    # Severe barrier impact frame: -18.0 m/s^2 deceleration
    f_impact = 5
    kin_impact = TrackKinematics(1, f_impact, f_impact * 100000000, (10, 10), (0, 0), 0.0, 0.0, (-18, 0), -18.0, 0, (1, 0), 0, None, True)
    cands = detector.evaluate(f_impact, f_impact * 100000000, "v1", {1: tr}, {1: kin_impact}, [], lm)

    assert len(cands) == 1
    cand = cands[0]
    assert cand.event_type in ("single_vehicle_anomaly", "barrier_strike")
    assert "severe_barrier_impact" in cand.signals["anomaly_type"]
    assert cand.primary_track_id == 1


def test_single_vehicle_rollover():
    detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=4,
        aspect_ratio_change_threshold=0.35,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr = Track(1, 2, "car")
    tr.track_age = 10
    tr.current_bbox_xywh = (200, 200, 80, 40)  # aspect ratio = 2.0
    tr.ground_history = [(10.0, 10.0)] * 10

    for f in range(5):
        kin = TrackKinematics(1, f, f * 100000000, (10, 10), (8, 0), 8.0, 28.8, (0, 0), 0.0, 0, (1, 0), 0, None, True)
        detector.evaluate(f, f * 100000000, "v1", {1: tr}, {1: kin}, [], lm)

    # Rollover frame: car flips onto side, w=40, h=80 -> aspect ratio = 0.5 (75% drop) + deceleration
    f_roll = 5
    tr.current_bbox_xywh = (200, 200, 40, 80)
    kin_roll = TrackKinematics(1, f_roll, f_roll * 100000000, (10, 10), (2, 0), 2.0, 7.2, (-6, 0), -6.0, 0, (1, 0), 0, None, True)
    cands = detector.evaluate(f_roll, f_roll * 100000000, "v1", {1: tr}, {1: kin_roll}, [], lm)

    assert len(cands) == 1
    assert "rollover" in cands[0].signals["anomaly_type"]


def test_single_vehicle_abnormal_yaw_spin():
    detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=4,
        yaw_rate_threshold_deg=30.0,
        min_persistence_frames=1,
    )
    lm = LaneManager()

    tr = Track(1, 2, "car")
    tr.track_age = 10
    tr.current_bbox_xywh = (200, 200, 80, 50)
    tr.ground_history = [(10.0, 10.0)] * 10

    for f in range(5):
        kin = TrackKinematics(1, f, f * 100000000, (10, 10), (8, 0), 8.0, 28.8, (0, 0), 0.0, 0.0, (1, 0), 0, None, True)
        detector.evaluate(f, f * 100000000, "v1", {1: tr}, {1: kin}, [], lm)

    # Sudden 60-degree spin-out with deceleration
    f_spin = 5
    kin_spin = TrackKinematics(1, f_spin, f_spin * 100000000, (10, 10), (3, 5), 5.8, 20.0, (-6, 0), -6.0, 60.0, (0.5, 0.86), 0, None, True)
    cands = detector.evaluate(f_spin, f_spin * 100000000, "v1", {1: tr}, {1: kin_spin}, [], lm)

    assert len(cands) == 1
    assert "yaw" in cands[0].signals["anomaly_type"]


def test_single_vehicle_normal_driving_does_not_trigger():
    detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=4,
    )
    lm = LaneManager()

    tr = Track(1, 2, "car")
    tr.track_age = 10
    tr.current_bbox_xywh = (200, 200, 80, 50)
    tr.ground_history = [(10.0, 10.0)] * 10

    # Normal cruising and mild braking (-2.0 m/s^2)
    for f in range(10):
        acc = -2.0 if f >= 5 else 0.0
        kin = TrackKinematics(1, f, f * 100000000, (10 + f, 10), (8, 0), 8.0, 28.8, (acc, 0), acc, 0.0, (1, 0), 0, None, True)
        cands = detector.evaluate(f, f * 100000000, "v1", {1: tr}, {1: kin}, [], lm)
        assert len(cands) == 0


def test_single_vehicle_edge_clipping_rejected():
    detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=True,
        video_warmup_frames=0,
        min_track_age=4,
        edge_margin_px=20.0,
    )
    lm = LaneManager()

    tr = Track(1, 2, "car")
    tr.track_age = 10
    tr.current_bbox_xywh = (5, 200, 80, 50)  # Clipped by left edge (x=5 < 20)
    tr.ground_history = [(10.0, 10.0)] * 10

    for f in range(5):
        kin = TrackKinematics(1, f, f * 100000000, (10, 10), (10, 0), 10.0, 36.0, (0, 0), 0.0, 0, (1, 0), 0, None, True)
        detector.evaluate(f, f * 100000000, "v1", {1: tr}, {1: kin}, [], lm)

    # Severe impact at edge should be rejected due to edge clipping
    kin_edge = TrackKinematics(1, 5, 500000000, (10, 10), (0, 0), 0.0, 0.0, (-20, 0), -20.0, 0, (1, 0), 0, None, True)
    cands = detector.evaluate(5, 500000000, "v1", {1: tr}, {1: kin_edge}, [], lm)
    assert len(cands) == 0
