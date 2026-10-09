"""Unit tests for modules/vocabulary_gate.py (Iteration K.3)."""

import pytest
import numpy as np
from road_surveillance.calibration.homography import PlanarHomography
from modules.vocabulary_gate import (
    VocabularyGate,
    compute_bev_footprint,
    is_point_in_polygon,
    VEHICLE_SUPERCLASS,
)


@pytest.fixture
def calibrated_homography():
    src = [[0.0, 720.0], [1280.0, 720.0], [850.0, 300.0], [430.0, 300.0]]
    dst = [[-10.0, 0.0], [10.0, 0.0], [10.0, 50.0], [-10.0, 50.0]]
    return PlanarHomography(
        source_points=src,
        destination_points=dst,
        calib_resolution=(1280, 720),
        target_resolution=(640, 360),
    )


def test_compute_bev_footprint_vehicle(calibrated_homography):
    # Standard car box at bottom of road (x1=300, y1=160, x2=350, y2=200)
    box = np.array([300.0, 160.0, 350.0, 200.0])
    w_m, l_m, area_m2 = compute_bev_footprint(box, calibrated_homography)

    assert w_m >= 1.4
    assert l_m >= 2.2
    assert area_m2 >= 1.5


def test_debris_discrimination_not_promoted(calibrated_homography):
    # Small debris box (e.g. 8x6 pixels)
    gate = VocabularyGate(homography=calibrated_homography)
    debris_box = [320.0, 195.0, 328.0, 201.0]

    raw_dets = [
        {
            "bbox": debris_box,
            "conf": 0.28,
            "class_name": "bottle",
            "class_id": 39,
            "secondary_vehicle_conf": 0.08,
        }
    ]

    audited = gate.audit_and_reclassify_detections(raw_dets, (360, 640))
    det = audited[0]

    # Debris must NOT be reclassified as a vehicle
    assert not det["audit_reclassified"]
    assert not det["is_vehicle"]
    assert det["class_name"] == "bottle"


def test_large_obstacle_recovered_as_vehicle(calibrated_homography):
    # Large obstacle on highway travel lanes misclassified as suitcase/bottle
    gate = VocabularyGate(homography=calibrated_homography)
    car_sized_box = [300.0, 160.0, 350.0, 200.0]

    raw_dets = [
        {
            "bbox": car_sized_box,
            "conf": 0.35,
            "class_name": "suitcase",
            "class_id": 28,
            "secondary_vehicle_conf": 0.09,
        }
    ]

    audited = gate.audit_and_reclassify_detections(raw_dets, (360, 640))
    det = audited[0]

    # Large vehicle-sized detection with secondary vehicle score must be recovered!
    assert det["audit_reclassified"]
    assert det["is_vehicle"]
    assert det["class_name"] == "candidate_vehicle_recovered"
    assert det["fused_vehicle_conf"] == 0.09


def test_offroad_detection_not_promoted(calibrated_homography):
    # Detection far off to the left or in the sky (above horizon)
    gate = VocabularyGate(homography=calibrated_homography)
    sky_box = [50.0, 20.0, 120.0, 60.0]

    raw_dets = [
        {
            "bbox": sky_box,
            "conf": 0.40,
            "class_name": "chair",
            "class_id": 56,
            "secondary_vehicle_conf": 0.12,
        }
    ]

    audited = gate.audit_and_reclassify_detections(raw_dets, (360, 640))
    det = audited[0]

    # Off-road non-vehicle must NOT be reclassified as vehicle
    assert not det["audit_reclassified"]
    assert not det["is_vehicle"]
    assert not det["in_roadway"]
