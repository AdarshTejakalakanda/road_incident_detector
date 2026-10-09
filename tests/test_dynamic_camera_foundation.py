"""Unit tests for Dynamic Camera, Homography Foundation, and Horizon Singularity Protection."""

import pytest
import numpy as np
from road_surveillance.video.metadata import VideoMetadata, probe_video_metadata
from road_surveillance.calibration.camera_config import CameraConfiguration, CameraCalibrationMode
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.tracking.track_state import Track, TrackletState


def test_video_metadata_probing():
    """Verify video metadata extraction and properties."""
    meta = VideoMetadata(
        video_id="test_vid",
        camera_id="cam_01",
        resolution=(1920, 1080),
        fps=25.0,
        total_frames=100,
        duration_sec=4.0,
        file_path="/tmp/test.mp4",
    )
    assert meta.width == 1920
    assert meta.height == 1080
    assert pytest.approx(meta.aspect_ratio, rel=1e-3) == 16.0 / 9.0
    assert meta.fps == 25.0
    assert meta.duration_sec == 4.0


def test_dynamic_homography_no_hardcoding():
    """Verify homography adapts dynamically between arbitrary resolutions without 1280x720 hardcoding."""
    # Define reference points on arbitrary 800x600 resolution
    ref_src = [[0.0, 600.0], [800.0, 600.0], [600.0, 200.0], [200.0, 200.0]]
    ref_dst = [[-10.0, 0.0], [10.0, 0.0], [10.0, 40.0], [-10.0, 40.0]]

    h_ref = PlanarHomography(
        source_points=ref_src,
        destination_points=ref_dst,
        calib_resolution=(800, 600),
        is_calibrated=True,
    )

    # Adapt to 1600x1200 (scale factor exactly 2.0x)
    h_1600 = h_ref.adapt_to_resolution((1600, 1200))
    assert h_1600.target_resolution == (1600, 1200)

    # A point at (400, 400) in 800x600 corresponds to (800, 800) in 1600x1200
    gx_ref, gy_ref = h_ref.pixel_to_ground(400.0, 400.0)
    gx_scaled, gy_scaled = h_1600.pixel_to_ground(800.0, 800.0)

    assert pytest.approx(gx_scaled, abs=1e-2) == gx_ref
    assert pytest.approx(gy_scaled, abs=1e-2) == gy_ref


def test_horizon_singularity_detection_and_rejection():
    """Verify vanishing line horizon singularities are rejected and do not produce NaN or runaway coordinates."""
    # Source trapezoid with a distinct vanishing line
    src = [[0.0, 720.0], [1280.0, 720.0], [850.0, 300.0], [430.0, 300.0]]
    dst = [[-10.0, 0.0], [10.0, 0.0], [10.0, 50.0], [-10.0, 50.0]]
    h = PlanarHomography(source_points=src, destination_points=dst, calib_resolution=(1280, 720))

    # Points on the road surface (e.g. bottom center) must be valid
    assert h.is_valid_ground_point(640.0, 650.0) is True

    # Vanishing line check: points far into the sky (e.g. v = -500 or near horizon)
    horizon_line = h.horizon_line
    assert len(horizon_line) == 3

    # For points with W' <= 0 or extreme vanishing proximity, is_valid_ground_point must return False
    # Check top edge of frame / sky
    assert h.is_valid_ground_point(640.0, -100.0) is False

    # pixel_to_ground_validated returns is_valid=False for points outside road / above horizon
    x, y, valid = h.pixel_to_ground_validated(640.0, -200.0)
    assert valid is False


def test_explicit_calibration_modes():
    """Verify explicit handling of calibrated vs uncalibrated modes."""
    cfg_uncalib = CameraConfiguration.create_uncalibrated("cam_test")
    assert cfg_uncalib.is_calibrated is False
    assert cfg_uncalib.mode == CameraCalibrationMode.UNCALIBRATED

    cfg_calib = CameraConfiguration.create_calibrated(
        camera_id="cam_calib",
        source_points=[[0.0, 100.0], [100.0, 100.0], [100.0, 0.0], [0.0, 0.0]],
        destination_points=[[0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
        calib_resolution=(100, 100),
    )
    assert cfg_calib.is_calibrated is True
    assert cfg_calib.mode == CameraCalibrationMode.CALIBRATED


def test_physical_jump_and_speed_clamping():
    """Verify GeometryEngine clamps impossible speeds and coordinates."""
    h = PlanarHomography(is_calibrated=True)
    engine = GeometryEngine(homography=h, is_camera_calibrated=True)

    tr = Track(track_id=1, class_id=2, class_label="car")
    tr.position_history = [(100.0, 100.0)]
    tr.ground_history = [(0.0, 0.0)]
    tr.timestamps = [0]

    # Frame 1: reasonable movement (1 meter in 0.1s = 10 m/s = 36 km/h)
    tr.position_history.append((105.0, 100.0))
    tr.timestamps.append(int(0.1 * 1e9))
    tlet1 = TrackletState("v1", 1, 2, "car", 1, int(0.1 * 1e9), (105, 100, 50, 40), (1.0, 0.0), "active")
    kin_map, _ = engine.process_frame([tlet1], {1: tr})
    assert kin_map[1].speed_kmh <= 220.0
    assert kin_map[1].is_calibrated is True

    # Frame 2: catastrophic jump (1,000 meters in 0.033s = 30,000 m/s)
    tr.position_history.append((200.0, 100.0))
    tr.timestamps.append(int(0.133 * 1e9))
    tlet2 = TrackletState("v1", 1, 2, "car", 2, int(0.133 * 1e9), (200, 100, 50, 40), (1000.0, 0.0), "active")
    kin_map2, _ = engine.process_frame([tlet2], {1: tr})

    # Speed must be clamped to MAX_ROAD_SPEED_KMH (220 km/h)
    assert kin_map2[1].speed_kmh <= 220.0
    assert abs(kin_map2[1].acceleration_mag) <= 60.0


def test_plane_of_motion_pixel_kinematics():
    """Verify invariant image-space pixel velocity and acceleration tracking."""
    h = PlanarHomography(is_calibrated=False)
    engine = GeometryEngine(homography=h, is_camera_calibrated=False)

    tr = Track(track_id=2, class_id=2, class_label="car")
    tr.position_history = [(200.0, 300.0)]
    tr.ground_history = [(0.0, 0.0)]
    tr.timestamps = [0]

    # Advance 30 frames moving 10 pixels per frame at 30 fps
    fps = 30.0
    dt_ns = int((1.0 / fps) * 1e9)
    for i in range(1, 5):
        pts = i * dt_ns
        cur_x = 200.0 + i * 10.0
        tr.position_history.append((cur_x, 300.0))
        tr.timestamps.append(pts)
        tlet = TrackletState("v", 2, 2, "car", i, pts, (cur_x, 300.0, 60.0, 40.0), (cur_x, 340.0), "active")
        kin_map, _ = engine.process_frame([tlet], {2: tr})

    # Pixel speed = 10 px / (1/30 s) = 300 px/sec
    assert pytest.approx(kin_map[2].pixel_speed, rel=1e-2) == 300.0
    assert kin_map[2].is_calibrated is False
