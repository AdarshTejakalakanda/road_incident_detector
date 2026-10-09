"""Unit tests for modules/detector_quality_gate.py (Iteration K.4)."""

import pytest
import numpy as np
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.tracking import RobustSurveillanceTracker
from modules.detector_quality_gate import DetectionRecord, DetectorQualityGate


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


def test_high_conf_admitted_anywhere(calibrated_homography):
    gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10)
    raw_dets = [
        {"bbox": [10.0, 20.0, 60.0, 80.0], "conf": 0.45, "class_name": "car", "class_id": 2}
    ]
    admitted, tel = gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=1,
        timestamp_s=0.033,
        frame_shape=(360, 640),
        homography=calibrated_homography,
    )
    assert len(admitted) == 1
    assert tel["high_conf_admitted"] == 1
    assert admitted[0].raw_confidence == 0.45


def test_low_conf_roadway_rescue(calibrated_homography):
    gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10)
    # Car box on active roadway with conf = 0.15
    car_box = [300.0, 160.0, 350.0, 200.0]
    raw_dets = [
        {"bbox": car_box, "conf": 0.15, "class_name": "car", "class_id": 2}
    ]
    admitted, tel = gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=1,
        timestamp_s=0.033,
        frame_shape=(360, 640),
        homography=calibrated_homography,
    )
    assert len(admitted) == 1
    assert tel["low_conf_rescued"] == 1
    assert admitted[0].is_on_roadway
    assert admitted[0].trust_score > 0.50


def test_low_conf_offroad_rejected(calibrated_homography):
    gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10)
    # Offroad box (above horizon) with conf = 0.18
    sky_box = [20.0, 10.0, 80.0, 50.0]
    raw_dets = [
        {"bbox": sky_box, "conf": 0.18, "class_name": "car", "class_id": 2}
    ]
    admitted, tel = gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=1,
        timestamp_s=0.033,
        frame_shape=(360, 640),
        homography=calibrated_homography,
    )
    assert len(admitted) == 0
    assert tel["rejected_offroad_low"] == 1


def test_low_conf_debris_geometry_rejected(calibrated_homography):
    gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10)
    # Tiny box on roadway with conf = 0.18 (fails metric ground threshold Wm >= 1.2, Lm >= 1.8)
    tiny_box = [320.0, 195.0, 326.0, 200.0]
    raw_dets = [
        {"bbox": tiny_box, "conf": 0.18, "class_name": "car", "class_id": 2}
    ]
    admitted, tel = gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=1,
        timestamp_s=0.033,
        frame_shape=(360, 640),
        homography=calibrated_homography,
    )
    assert len(admitted) == 0
    assert tel["rejected_geometry"] == 1


def test_trust_score_formulation():
    gate = DetectorQualityGate()
    t_high = gate.compute_trust_score(raw_confidence=0.40, is_on_roadway=True, area_ratio=0.003)
    assert pytest.approx(t_high, abs=1e-3) == 1.0

    t_mid = gate.compute_trust_score(raw_confidence=0.15, is_on_roadway=True, area_ratio=0.002)
    # w1 * (0.15 / 0.4) + w2 * 1.0 + w3 * 1.0 = 0.5 * 0.375 + 0.35 + 0.15 = 0.1875 + 0.50 = 0.6875
    assert 0.50 <= t_mid <= 0.70


def test_tracker_ingestion_with_detection_records():
    tracker = RobustSurveillanceTracker()
    rec = DetectionRecord(
        detection_id=1,
        source_frame=1,
        timestamp_s=0.033,
        raw_class="car",
        fused_class="vehicle",
        raw_confidence=0.85,
        trust_score=0.95,
        bbox_xyxy=(100.0, 100.0, 200.0, 200.0),
        bbox_area_ratio=0.005,
        is_on_roadway=True,
        ground_dimensions_m=(2.0, 4.5),
        audit_reclassified=False,
    )

    tracklets, tracks = tracker.update([rec])
    assert len(tracks) == 1
    t = list(tracks.values())[0]
    assert pytest.approx(t.trust_score, abs=1e-3) == 0.95
    assert len(tracklets) == 1
    assert pytest.approx(tracklets[0].trust_score, abs=1e-3) == 0.95
    assert tracklets[0].provenance == "REAL"


def test_patch_n1_roadway_traffic_light_recovery(calibrated_homography):
    """Verify Patch N.1 3-condition geometric gate for roadway traffic_light recovery."""
    gate = DetectorQualityGate()

    # 1. Valid roadway vehicle misclassified as traffic_light:
    # Bbox on roadway (y > horizon), footprint Wm ~ 1.6m, Lm ~ 5.6m
    valid_tl_box = [300.0, 320.0, 340.0, 340.0]

    # 2. Overhead / sky traffic light:
    sky_tl_box = [200.0, 20.0, 240.0, 80.0]

    # 3. Off-roadside pole with huge longitudinal extent:
    roadside_fixture_box = [500.0, 150.0, 520.0, 300.0]

    raw_dets = [
        {"bbox": valid_tl_box, "conf": 0.035, "class_name": "traffic_light", "class_id": 9},
        {"bbox": sky_tl_box, "conf": 0.45, "class_name": "traffic_light", "class_id": 9},
        {"bbox": roadside_fixture_box, "conf": 0.15, "class_name": "traffic_light", "class_id": 9},
    ]

    admitted, tel = gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=84,
        timestamp_s=2.80,
        frame_shape=(360, 640),
        homography=calibrated_homography,
    )

    # Valid roadway traffic_light must be recovered as candidate_vehicle_recovered
    cand_records = [r for r in admitted if r.fused_class == "candidate_vehicle_recovered"]
    assert len(cand_records) == 1
    rec = cand_records[0]
    assert rec.raw_class == "traffic_light"
    assert rec.fused_class == "candidate_vehicle_recovered"
    assert rec.audit_reclassified is True
    assert rec.original_class == "traffic_light"
    assert rec.recovery_reason == "roadway_semantic_remap"
    assert rec.is_on_roadway is True

    # Overhead traffic light must NOT be admitted as candidate_vehicle_recovered
    sky_records = [r for r in admitted if r.raw_class == "traffic_light" and not r.is_on_roadway]
    assert all(r.fused_class != "candidate_vehicle_recovered" for r in sky_records)
