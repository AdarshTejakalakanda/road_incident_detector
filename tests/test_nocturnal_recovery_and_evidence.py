"""Tests for Iteration J.1 Nocturnal Recovery and Iteration H.1 Zoomed Evidence."""

import os
import numpy as np
import pytest

from road_surveillance.video import IngestedFrame
from road_surveillance.detection.yolo_detector import YOLO26Detector
from road_surveillance.calibration.road_surface import RoadSurfaceDetector
from road_surveillance.verification.evidence_extractor import CandidateEvidenceExtractor


def test_yolo_nocturnal_class_filtering():
    """Verify reduced threshold applies only to vehicles and remapped headlights."""
    detector = YOLO26Detector(
        conf_threshold=0.25,
        nocturnal_conf_threshold=0.10,
        enable_nocturnal_headlight_remap=True,
    )
    assert detector.nocturnal_conf_threshold == 0.10
    assert "car" in detector.VEHICLE_CLASSES
    assert "traffic light" in detector.HEADLIGHT_CLASSES


def test_evidence_extractor_package_retention():
    """Verify evidence extractor retains full frames, 384x384 crops, temporal frames, and SoM."""
    extractor = CandidateEvidenceExtractor(
        roi_target_size=(384, 384),
        roi_expansion_factor=2.0,
    )

    # Provide synthetic frames in cache
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cached = {f: dummy_img for f in [0, 15, 30, 45, 60]}

    pkg = extractor.extract_evidence_package(
        video_id="dummy_video",
        candidate_frame=30,
        candidate_detector="collision",
        involved_track_ids=[1, 2],
        track_boxes={1: (600, 300, 80, 60), 2: (670, 310, 80, 60)},
        cached_frames=cached,
    )

    # 1. Full frames retained
    assert "t0" in pkg.full_frames
    assert len(pkg.full_frames) == 5

    # 2. 384x384 focused ROI crops
    assert "t0" in pkg.roi_crops
    assert pkg.roi_crops["t0"].shape == (384, 384, 3)

    # 3. Temporal ROI frames (5 frames)
    assert list(pkg.roi_crops.keys()) == ["t-2", "t-1", "t0", "t+1", "t+2"]

    # 4. Set-of-Mark crops
    assert "t0" in pkg.som_crops
    assert pkg.som_crops["t0"].shape == (384, 384, 3)

    # 5. ROI bounding box
    rx, ry, rw, rh = pkg.roi_bbox
    assert rw > 0 and rh > 0
