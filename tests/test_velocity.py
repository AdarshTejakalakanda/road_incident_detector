"""Unit tests for Velocity and Speed Calculation."""

import pytest
from road_surveillance.geometry import GeometryEngine
from road_surveillance.calibration import PlanarHomography
from road_surveillance.tracking import TrackletState, Track


def test_velocity_and_speed_calculation():
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)

    track = Track(track_id=1, class_id=2, class_label="car")

    # Frame 0: at (0, 0)
    t0 = TrackletState("v1", 1, 2, "car", 0, 0, (0, 0, 10, 10), (0.0, 0.0), "active")
    track.update((0, 0, 10, 10), 0.9, 0)
    kin0, _ = engine.process_frame([t0], {1: track})

    assert kin0[1].speed_kmh == 0.0

    # Frame 1: moved by 5.0m in 1.0 second (1e9 ns) -> dx=5.0m (< max_jump 8m)
    # EMA filtered dx = 5.0 * 0.6 = 3.0m, dt=1s -> vx=3.0 m/s -> 10.8 km/h
    t1 = TrackletState("v1", 1, 2, "car", 1, int(1e9), (5, 0, 10, 10), (5.0, 0.0), "active")
    track.update((5, 0, 10, 10), 0.9, int(1e9))
    kin1, _ = engine.process_frame([t1], {1: track})

    assert kin1[1].velocity_vector[0] > 0.0
    assert pytest.approx(kin1[1].speed_mps, rel=1e-1) == 3.0
    assert pytest.approx(kin1[1].speed_kmh, rel=1e-1) == 10.8


def test_velocity_varying_fps_dt():
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=2, class_id=2, class_label="car")

    # Variable dt: 0.5 second (5e8 ns)
    t0 = TrackletState("v1", 2, 2, "car", 0, 0, (0, 0, 10, 10), (0.0, 0.0), "active")
    track.update((0, 0, 10, 10), 0.9, 0)
    engine.process_frame([t0], {2: track})

    t1 = TrackletState("v1", 2, 2, "car", 1, int(5e8), (5, 0, 10, 10), (5.0, 0.0), "active")
    track.update((5, 0, 10, 10), 0.9, int(5e8))
    kin1, _ = engine.process_frame([t1], {2: track})

    # Filtered pos: 0.6 * 5 = 3.0 m, dt = 0.5 s -> v = 6.0 m/s
    assert pytest.approx(kin1[2].speed_mps, rel=1e-1) == 6.0
