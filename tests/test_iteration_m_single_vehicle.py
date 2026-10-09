"""Unit tests for Iteration M: Single-Vehicle Kinematic Recovery & Hypothesis Arbitration."""

import pytest
import numpy as np
from modules.single_vehicle import (
    calculate_heading_deviation,
    distance_to_polygon_boundary,
    check_has_lead_vehicle,
    compute_single_vehicle_evidence,
)
from modules.vlm_arbitrator import GroqArbitrator, SINGLE_VEHICLE_PROMPT
from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics


def test_calculate_heading_deviation_straight():
    positions = [(0.0, float(i * 10.0)) for i in range(10)]
    theta = calculate_heading_deviation(positions, k=5)
    assert theta == pytest.approx(0.0, abs=1e-2)


def test_calculate_heading_deviation_right_angle():
    # Past segment going north (0,0)->(0,20), current segment going east (0,20)->(20,20)
    positions = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (10.0, 20.0), (20.0, 20.0)]
    theta = calculate_heading_deviation(positions, k=4)
    assert theta == pytest.approx(90.0, abs=1.0)


def test_calculate_heading_deviation_spin_out():
    # Vehicle moving forward, then spins backwards (180 deg reversal)
    positions = [(0.0, 0.0), (0.0, 10.0), (0.0, 20.0), (0.0, 10.0), (0.0, 0.0)]
    theta = calculate_heading_deviation(positions, k=4)
    assert theta == pytest.approx(180.0, abs=1.0)


def test_calculate_heading_deviation_stationary():
    positions = [(10.0, 10.0)] * 6
    theta = calculate_heading_deviation(positions, k=5)
    assert theta == 0.0


def test_distance_to_polygon_boundary():
    # Roadway box from x=0..10, y=0..100
    poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 100.0), (0.0, 100.0)]
    
    # Point at x=0.2 (0.2m from left boundary)
    d1 = distance_to_polygon_boundary((0.2, 50.0), poly)
    assert d1 == pytest.approx(0.2, abs=0.05)

    # Point centered at x=5.0 (5m from left and right boundaries)
    d2 = distance_to_polygon_boundary((5.0, 50.0), poly)
    assert d2 == pytest.approx(5.0, abs=0.05)


def test_check_has_lead_vehicle():
    tr1 = Track(1, 2, "car")
    tr1.ground_history = [(0.0, 10.0)]
    
    # Vehicle 2 is 10m directly ahead of vehicle 1
    tr2 = Track(2, 2, "car")
    tr2.ground_history = [(0.0, 20.0)]
    
    kin1 = TrackKinematics(1, 0, 0, (0.0, 10.0), (0, 10), 10.0, 36.0, (0, 0), 0.0, 90.0, (0.0, 1.0), 0, None, True)
    kin2 = TrackKinematics(2, 0, 0, (0.0, 20.0), (0, 10), 10.0, 36.0, (0, 0), 0.0, 90.0, (0.0, 1.0), 0, None, True)

    tracks = {1: tr1, 2: tr2}
    kinematics = {1: kin1, 2: kin2}

    has_lead = check_has_lead_vehicle(1, tracks, kinematics, max_forward_dist_m=25.0)
    assert has_lead is True

    # Vehicle 2 has no lead vehicle ahead
    has_lead_2 = check_has_lead_vehicle(2, tracks, kinematics, max_forward_dist_m=25.0)
    assert has_lead_2 is False


def test_compute_single_vehicle_evidence_spin_and_impact():
    poly = [(0.0, 0.0), (12.0, 0.0), (12.0, 200.0), (0.0, 200.0)]

    tr = Track(1, 2, "car")
    # Vehicle heading north, spins 60 deg towards edge
    tr.position_history = [(6.0, 0.0), (6.0, 10.0), (6.0, 20.0), (3.0, 25.0), (0.5, 27.0)]
    tr.ground_history = tr.position_history
    tr.bbox_history = [(100, 100, 80, 40)] * 5
    tr.current_bbox_xywh = (100, 100, 80, 40)

    # Deceleration of -6.0 m/s^2 (braking/stopping)
    kin = TrackKinematics(1, 5, 0, (0.5, 27.0), (0, 0), 0.0, 0.0, (0, -6), -6.0, 0, (0, 1), 0, None, True)

    score, breakdown = compute_single_vehicle_evidence(
        track=tr,
        roadway_poly=poly,
        window_k=4,
        kinematics=kin,
        has_lead_vehicle=False,
    )

    # Expect high angular deviation (s_theta > 0.8), high decel (s_decel = 1.0), close to edge (s_bound = 1.0)
    assert breakdown["s_decel"] == pytest.approx(1.0, abs=0.01)
    assert breakdown["s_bound"] == pytest.approx(1.0, abs=0.01)
    assert score >= 0.65


def test_compute_single_vehicle_evidence_normal_driving():
    poly = [(0.0, 0.0), (12.0, 0.0), (12.0, 200.0), (0.0, 200.0)]

    tr = Track(1, 2, "car")
    # Straight steady motion in center of lane
    tr.position_history = [(6.0, float(i * 10.0)) for i in range(10)]
    tr.ground_history = tr.position_history
    tr.bbox_history = [(100, 100, 60, 30)] * 10
    tr.current_bbox_xywh = (100, 100, 60, 30)

    # Steady speed, zero acceleration
    kin = TrackKinematics(1, 10, 0, (6.0, 90.0), (0, 10), 10.0, 36.0, (0, 0), 0.0, 90.0, (0, 1), 0, None, True)

    score, breakdown = compute_single_vehicle_evidence(
        track=tr,
        roadway_poly=poly,
        window_k=5,
        kinematics=kin,
        has_lead_vehicle=False,
    )

    assert breakdown["s_theta"] == 0.0
    assert breakdown["s_decel"] == 0.0
    assert breakdown["s_bound"] == 0.0
    assert score == 0.0


def test_groq_arbitrator_single_vehicle_fallback():
    arb = GroqArbitrator(mock_fallback=True)

    telemetry = {
        "event_type": "single_vehicle_anomaly",
        "composite_score": 0.72,
        "accel_ms2": -8.5,
        "yaw_delta_deg": 45.0,
        "aspect_ratio_ratio": 0.25,
        "d_edge_m": 0.3,
        "anomaly_type": "composite(0.72)",
        "track_age": 12,
        # Speed field required: genuine incidents come from moving vehicles (>=15 km/h gate)
        "max_recent_speed_kmh": 45.0,
    }

    dummy_strip = np.zeros((336, 600, 3), dtype=np.uint8)
    verdict = arb.arbitrate(dummy_strip, telemetry, candidate_frame=60, video_id="test_single_00")

    assert verdict["verdict"] == "CANDIDATE_UNCONFIRMED"
    assert verdict["suppressed"] is True
    assert verdict["source"] == "offline_unverified_fallback"
