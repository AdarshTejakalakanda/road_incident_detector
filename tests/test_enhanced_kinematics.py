"""Unit tests for the Enhanced Kinematics Subsystem.

Validates:
1. Multi-frame windowed regression canceling discrete pixel quantization jitter
2. Tangential vs Centripetal acceleration decomposition (a_long vs a_lat)
3. Heading stabilization and low-speed yaw rate deadband
4. Stationary vehicle zero-jitter acceleration deadband
5. Physical barrier crash rapid deceleration detection
"""

import math
import pytest
import numpy as np
from road_surveillance.geometry import GeometryEngine
from road_surveillance.calibration import PlanarHomography
from road_surveillance.tracking import TrackletState, Track


def test_smoothed_velocity_cancels_pixel_jitter():
    """Verify that alternating +/- 0.15m bounding box jitter does not produce false velocity swings."""
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=1, class_id=2, class_label="car")

    # Vehicle cruising steadily at 15 m/s (~54 km/h) with +/- 0.15m detector coordinate jitter
    base_speed = 15.0
    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)
    kin_list = []

    for f in range(12):
        t_ns = f * dt_ns
        jitter = 0.15 if (f % 2 == 1) else -0.15
        true_x = f * base_speed * dt
        meas_x = true_x + jitter

        t_state = TrackletState("v1", 1, 2, "car", f, t_ns, (meas_x, 0, 10, 10), (meas_x, 0.0), "active")
        track.update((meas_x, 0, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {1: track})
        kin_list.append(kin_map[1])

    # After warm-up (f >= 5), smoothed speed should stay close to true 15 m/s (54 km/h)
    # without pegging at extreme values
    mature_speeds = [k.speed_kmh for k in kin_list[5:]]
    for spd in mature_speeds:
        # Expected around 54 km/h +/- 20%
        assert 35.0 <= spd <= 75.0, f"Speed swing out of bounds: {spd} km/h"

    # Crucially, longitudinal acceleration should NOT false-trigger emergency braking (<= -14 m/s^2)
    mature_accels = [k.acceleration_longitudinal for k in kin_list[5:]]
    for acc in mature_accels:
        assert acc > -14.0, f"False severe deceleration spike detected: {acc} m/s^2"


def test_longitudinal_vs_lateral_acceleration_decomposition():
    """Verify that turning/centripetal acceleration creates a_lat, not false negative braking a_long."""
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=2, class_id=2, class_label="car")

    # Vehicle moving in a circle of radius R=20m at constant speed v=10 m/s
    # Centripetal acceleration a_lat = v^2 / R = 100 / 20 = 5.0 m/s^2
    # Longitudinal acceleration a_long should be ~0.0 m/s^2 (constant speed)
    v_target = 10.0
    R = 20.0
    omega = v_target / R  # 0.5 rad/s
    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)
    kin_list = []

    for f in range(20):
        t_ns = f * dt_ns
        t_sec = f * dt
        angle = omega * t_sec
        x = R * math.cos(angle)
        y = R * math.sin(angle)

        t_state = TrackletState("v1", 2, 2, "car", f, t_ns, (x, y, 10, 10), (x, y), "active")
        track.update((x, y, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {2: track})
        kin_list.append(kin_map[2])

    # Check mature frames
    mature_k = kin_list[-5:]
    for k in mature_k:
        # a_lat should be positive significant centripetal acceleration
        assert abs(k.acceleration_lateral) > 1.0, f"Expected lateral acceleration during turn, got {k.acceleration_lateral}"
        # a_long should NOT be flagged as severe emergency braking (<= -10 m/s^2)
        assert k.acceleration_longitudinal > -10.0, f"False severe braking during turn: {k.acceleration_longitudinal}"


def test_heading_stabilization_and_yaw_deadband():
    """Verify heading is locked and yaw rate is 0 when vehicle is stopped, preventing random flips."""
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=3, class_id=2, class_label="car")

    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)

    # 1. Drive forward along East (heading = 0 deg) at 10 m/s for 10 frames
    for f in range(10):
        t_ns = f * dt_ns
        x = f * 10.0 * dt
        y = 0.0
        t_state = TrackletState("v1", 3, 2, "car", f, t_ns, (x, y, 10, 10), (x, y), "active")
        track.update((x, y, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {3: track})

    last_moving_k = kin_map[3]
    assert pytest.approx(last_moving_k.heading_deg, abs=5.0) == 0.0

    # 2. Vehicle comes to complete stop at x=x_stop, with sub-pixel noise (+/- 0.02m)
    x_stop = 10 * 10.0 * dt
    for f in range(10, 20):
        t_ns = f * dt_ns
        noise_x = 0.02 if (f % 2 == 1) else -0.02
        noise_y = -0.02 if (f % 2 == 1) else 0.02
        x = x_stop + noise_x
        y = noise_y
        t_state = TrackletState("v1", 3, 2, "car", f, t_ns, (x, y, 10, 10), (x, y), "active")
        track.update((x, y, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {3: track})

        stopped_k = kin_map[3]
        if f >= 16:
            # Heading should be preserved near 0 deg (or 360 deg), NOT flipping to 90, 180, 270 deg
            ang_err = min(stopped_k.heading_deg, abs(360.0 - stopped_k.heading_deg))
            assert ang_err < 15.0
            # Yaw rate should be deadbanded to 0.0
            assert stopped_k.yaw_rate_deg_s == 0.0


def test_stationary_vehicle_zero_jitter_acceleration():
    """Verify stopped vehicle does not generate false negative deceleration spikes from jitter."""
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=4, class_id=2, class_label="car")

    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)

    # Stationary vehicle sitting at (50, 50) for 15 frames with coordinate jitter
    for f in range(15):
        t_ns = f * dt_ns
        jx = 0.03 * math.sin(f)
        jy = 0.03 * math.cos(f)
        x = 50.0 + jx
        y = 50.0 + jy
        t_state = TrackletState("v1", 4, 2, "car", f, t_ns, (x, y, 10, 10), (x, y), "active")
        track.update((x, y, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {4: track})
        k = kin_map[4]

        if f >= 3:
            assert k.speed_kmh < 3.0
            # Acceleration should be 0.0 or near-zero, NEVER severe decel
            assert k.acceleration_longitudinal == 0.0 or abs(k.acceleration_longitudinal) < 1.0
            assert k.stationary_duration_sec > 0.0


def test_genuine_barrier_impact_detection():
    """Verify that a genuine sudden velocity collapse is accurately captured as severe deceleration."""
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=5, class_id=2, class_label="car")

    dt = 1.0 / 30.0
    dt_ns = int(dt * 1e9)
    base_speed = 20.0  # 72 km/h

    # 1. Cruise fast for 10 frames
    for f in range(10):
        t_ns = f * dt_ns
        x = f * base_speed * dt
        y = 0.0
        t_state = TrackletState("v1", 5, 2, "car", f, t_ns, (x, y, 10, 10), (x, y), "active")
        track.update((x, y, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {5: track})

    assert kin_map[5].speed_kmh > 40.0

    # 2. Impact barrier: vehicle suddenly arrests at x_impact
    x_impact = 10 * base_speed * dt + 0.10
    decel_values = []
    for f in range(10, 14):
        t_ns = f * dt_ns
        t_state = TrackletState("v1", 5, 2, "car", f, t_ns, (x_impact, 0, 10, 10), (x_impact, 0.0), "active")
        track.update((x_impact, 0, 10, 10), 0.9, t_ns)
        kin_map, _ = engine.process_frame([t_state], {5: track})
        decel_values.append(kin_map[5].acceleration_longitudinal)

    # At least one post-impact frame must show severe deceleration (<= -10.0 m/s^2)
    assert min(decel_values) <= -10.0, f"Expected crash deceleration <= -10.0 m/s^2, got min {min(decel_values)}"
