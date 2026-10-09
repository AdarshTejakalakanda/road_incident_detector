"""Unit tests for RoadSurfaceDetector."""

import pytest
import numpy as np

from road_surveillance.calibration.road_surface import RoadSurfaceDetector


def test_road_surface_default_horizon():
    detector = RoadSurfaceDetector(default_horizon_ratio=0.25)
    frame_shape = (1080, 1920)

    # Upper 25% = y < 270 px is sky / non-road
    assert detector.get_horizon_y(1080) == pytest.approx(270.0)

    # Point in sky should be rejected
    assert detector.is_point_on_road(960, 100, frame_shape) is False

    # Point on road should be accepted
    assert detector.is_point_on_road(960, 500, frame_shape) is True
    assert detector.is_point_on_road(960, 1000, frame_shape) is True


def test_road_surface_filter_off_road_detections():
    detector = RoadSurfaceDetector(default_horizon_ratio=0.30)
    frame_shape = (1000, 1000)
    # Horizon is at y = 300

    boxes = np.array([
        [100, 50, 150, 120],    # bottom = 120 (above horizon: sky billboard)
        [400, 400, 500, 600],   # bottom = 600 (below horizon: road vehicle)
        [200, 800, 300, 950],   # bottom = 950 (below horizon: road vehicle)
    ], dtype=np.float32)
    confs = np.array([0.60, 0.85, 0.90], dtype=np.float32)
    cids = np.array([2, 2, 2], dtype=np.int32)
    labels = ["car", "car", "truck"]

    f_boxes, f_confs, f_cids, f_labels = detector.filter_detections_on_road(
        boxes, confs, cids, labels, frame_shape
    )

    assert len(f_boxes) == 2
    assert len(f_labels) == 2
    assert "truck" in f_labels
    assert f_boxes[0][1] == 400  # First kept box is the second input box


def test_road_surface_perspective_horizon_estimation():
    detector = RoadSurfaceDetector(min_road_samples=10, enable_dynamic_horizon=True)
    frame_shape = (1000, 1000)

    # Synthesize vehicles whose height scales with perspective depth: h(y) = 0.2 * (y - 200)
    # So vanishing horizon is at y = 200
    synth_boxes = []
    for bottom_y in range(300, 950, 50):
        h = 0.2 * (bottom_y - 200.0)
        w = h * 1.5
        x = 500.0 - w / 2.0
        y = bottom_y - h
        synth_boxes.append((x, y, w, h))

    detector.register_vehicle_observations(synth_boxes, frame_shape)
    est_horiz = detector.get_horizon_y(1000)

    # Estimated horizon should be near 200 px
    assert 150.0 <= est_horiz <= 250.0
