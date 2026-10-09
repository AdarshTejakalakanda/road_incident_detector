"""Unit tests for Stationary Duration Tracking."""

import pytest
from road_surveillance.geometry import GeometryEngine
from road_surveillance.calibration import PlanarHomography
from road_surveillance.tracking import TrackletState, Track


def test_stationary_duration_accumulation():
    h = PlanarHomography()
    engine = GeometryEngine(homography=h, stationary_speed_threshold_kmh=3.0)
    track = Track(track_id=1, class_id=2, class_label="car")

    # Frame 0: t=0s, stationary at (10, 10)
    t0 = TrackletState("v1", 1, 2, "car", 0, 0, (10, 10, 10, 10), (15.0, 20.0), "active")
    track.update((10, 10, 10, 10), 0.9, 0)
    kin0, _ = engine.process_frame([t0], {1: track})
    assert kin0[1].stationary_duration_sec == 0.0

    # Frame 1: t=1s (1e9 ns), still stationary at (10, 10)
    t1 = TrackletState("v1", 1, 2, "car", 1, int(1e9), (10, 10, 10, 10), (15.0, 20.0), "active")
    track.update((10, 10, 10, 10), 0.9, int(1e9))
    kin1, _ = engine.process_frame([t1], {1: track})
    assert pytest.approx(kin1[1].stationary_duration_sec) == 1.0

    # Frame 2: t=3s (3e9 ns), still stationary -> accumulation should be 3.0s
    t2 = TrackletState("v1", 1, 2, "car", 2, int(3e9), (10, 10, 10, 10), (15.0, 20.0), "active")
    track.update((10, 10, 10, 10), 0.9, int(3e9))
    kin2, _ = engine.process_frame([t2], {1: track})
    assert pytest.approx(kin2[1].stationary_duration_sec) == 3.0

    # Frame 3: t=4s (4e9 ns), vehicle drives away fast to (60, 10)
    t3 = TrackletState("v1", 1, 2, "car", 3, int(4e9), (60, 10, 10, 10), (65.0, 20.0), "active")
    track.update((60, 10, 10, 10), 0.9, int(4e9))
    kin3, _ = engine.process_frame([t3], {1: track})
    assert kin3[1].stationary_duration_sec == 0.0
