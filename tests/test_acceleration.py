"""Unit tests for Acceleration and Deceleration Calculation."""

import pytest
from road_surveillance.geometry import GeometryEngine
from road_surveillance.calibration import PlanarHomography
from road_surveillance.tracking import TrackletState, Track


def test_deceleration_detection():
    h = PlanarHomography()
    engine = GeometryEngine(homography=h)
    track = Track(track_id=1, class_id=2, class_label="car")

    # Frame 0 at t=0s
    t0 = TrackletState("v1", 1, 2, "car", 0, 0, (0, 0, 10, 10), (0.0, 0.0), "active")
    track.update((0, 0, 10, 10), 0.9, 0)
    engine.process_frame([t0], {1: track})

    # Frame 1 at t=1s, moving fast
    t1 = TrackletState("v1", 1, 2, "car", 1, int(1e9), (20, 0, 10, 10), (20.0, 0.0), "active")
    track.update((20, 0, 10, 10), 0.9, int(1e9))
    kin1, _ = engine.process_frame([t1], {1: track})

    # Frame 2 at t=2s, stopped abruptly (position stays at 20)
    t2 = TrackletState("v1", 1, 2, "car", 2, int(2e9), (20, 0, 10, 10), (20.0, 0.0), "active")
    track.update((20, 0, 10, 10), 0.9, int(2e9))
    kin2, _ = engine.process_frame([t2], {1: track})

    # Deceleration should be signed negative
    assert kin2[1].acceleration_mag < 0.0
