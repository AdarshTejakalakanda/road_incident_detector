"""Unit tests for Planar Homography transformation and validation."""

import pytest
import numpy as np
from road_surveillance.calibration.homography import PlanarHomography


def test_homography_identity_transform():
    h = PlanarHomography()
    assert np.allclose(h.H, np.eye(3))

    gx, gy = h.pixel_to_ground(100.0, 200.0)
    assert pytest.approx(gx) == 100.0
    assert pytest.approx(gy) == 200.0

    px, py = h.ground_to_pixel(100.0, 200.0)
    assert pytest.approx(px) == 100.0
    assert pytest.approx(py) == 200.0


def test_homography_from_points():
    src = [[0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]]
    dst = [[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]]
    h = PlanarHomography(source_points=src, destination_points=dst)

    # Transform center point (50, 50) -> should be (5, 5)
    gx, gy = h.pixel_to_ground(50.0, 50.0)
    assert pytest.approx(gx, abs=1e-3) == 5.0
    assert pytest.approx(gy, abs=1e-3) == 5.0

    # Invert back (5, 5) -> (50, 50)
    px, py = h.ground_to_pixel(5.0, 5.0)
    assert pytest.approx(px, abs=1e-3) == 50.0
    assert pytest.approx(py, abs=1e-3) == 50.0


def test_homography_batch_transform():
    src = [[0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]]
    dst = [[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]]
    h = PlanarHomography(source_points=src, destination_points=dst)

    pts_px = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float64)
    pts_g = h.batch_pixel_to_ground(pts_px)
    assert pytest.approx(pts_g[0, 0], abs=1e-3) == 0.0
    assert pytest.approx(pts_g[0, 1], abs=1e-3) == 10.0
    assert pytest.approx(pts_g[1, 0], abs=1e-3) == 10.0
    assert pytest.approx(pts_g[1, 1], abs=1e-3) == 0.0


def test_homography_validation():
    src = [[0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]]
    dst = [[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]]
    h = PlanarHomography(source_points=src, destination_points=dst)

    res = h.validate_calibration(src, dst)
    assert res["is_invertible"] is True
    assert res["reprojection_error_rms"] < 1e-4
