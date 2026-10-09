"""Unit tests for Phase 2 Multimodal Verification module."""

import base64
import numpy as np
import pytest

from road_surveillance.verification.schema import VerificationVerdict, IncidentVerificationResult
from road_surveillance.verification.base import CandidateEvidencePackage
from road_surveillance.verification.evidence_extractor import CandidateEvidenceExtractor
from road_surveillance.verification.qwen_verifier import QwenMultimodalVerifier


def test_verification_schema_serialization():
    res = IncidentVerificationResult(
        event_id="test_ev_01",
        video_id="video_123",
        candidate_frame=45,
        verdict=VerificationVerdict.CONFIRMED_INCIDENT,
        confidence=0.92,
        rationale="Severe front-end crumple detected between tracks 1 and 2.",
        detected_crash_type="t_bone",
        involved_tracks=[1, 2],
        telemetry_summary={"tca_sec": 0.4, "min_distance_m": 0.8},
    )

    assert res.is_confirmed is True
    assert res.is_rejected is False
    assert res.is_uncertain is False

    d = res.to_dict()
    assert d["verdict"] == "CONFIRMED_INCIDENT"
    assert d["confidence"] == 0.92

    reconstructed = IncidentVerificationResult.from_dict(d)
    assert reconstructed.event_id == res.event_id
    assert reconstructed.verdict == VerificationVerdict.CONFIRMED_INCIDENT
    assert reconstructed.is_confirmed is True


def test_evidence_extractor_base64_encoding():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:50, 20:50] = (255, 0, 0)

    extractor = CandidateEvidenceExtractor()
    b64_str = extractor.encode_image_to_base64(img)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0

    decoded = base64.b64decode(b64_str)
    assert len(decoded) > 0


def test_evidence_package_creation():
    extractor = CandidateEvidenceExtractor()
    dummy_frames = {
        10: np.zeros((50, 50, 3), dtype=np.uint8),
        25: np.ones((50, 50, 3), dtype=np.uint8) * 128,
        40: np.ones((50, 50, 3), dtype=np.uint8) * 255,
    }

    pkg = extractor.extract_evidence_package(
        video_id="test_vid",
        candidate_frame=25,
        candidate_detector="pairwise_collision",
        involved_track_ids=[1, 2],
        telemetry={"iou": 0.35, "tca": 0.2},
        track_boxes={1: (5, 5, 20, 20), 2: (15, 15, 20, 20)},
        cached_frames=dummy_frames,
    )

    assert pkg.candidate_frame == 25
    assert pkg.candidate_detector == "pairwise_collision"
    assert "trigger" in pkg.keyframes
    assert pkg.keyframe_indices["trigger"] == 25


def test_qwen_verifier_mock_mode():
    verifier = QwenMultimodalVerifier(
        mock_mode=True,
        mock_default_verdict=VerificationVerdict.REJECTED_FALSE_ALARM,
    )

    pkg = CandidateEvidencePackage(
        event_id="test_01",
        video_id="vid_01",
        candidate_frame=30,
        candidate_detector="single_vehicle",
        involved_track_ids=[3],
        keyframes={"trigger": np.zeros((10, 10, 3), dtype=np.uint8)},
        keyframe_indices={"trigger": 30},
    )

    res = verifier.verify(pkg)
    assert res.is_rejected is True
    assert res.verdict == VerificationVerdict.REJECTED_FALSE_ALARM


def test_qwen_verifier_parse_vlm_json_variants():
    verifier = QwenMultimodalVerifier(mock_mode=False)
    pkg = CandidateEvidencePackage(
        event_id="test_02",
        video_id="vid_02",
        candidate_frame=50,
        candidate_detector="distant_conflict",
        involved_track_ids=[4, 5],
        keyframes={},
        keyframe_indices={},
    )

    # 1. Clean JSON
    json_clean = '{"verdict": "CONFIRMED_INCIDENT", "confidence": 0.95, "detected_crash_type": "rear_end", "rationale": "Clear impact."}'
    r1 = verifier._parse_vlm_response(json_clean, pkg)
    assert r1.verdict == VerificationVerdict.CONFIRMED_INCIDENT
    assert r1.confidence == 0.95
    assert r1.detected_crash_type == "rear_end"

    # 2. Markdown fenced JSON
    json_fenced = '```json\n{"verdict": "REJECTED_FALSE_ALARM", "confidence": 0.88, "detected_crash_type": "none", "rationale": "Vehicles cleared without contact."}\n```'
    r2 = verifier._parse_vlm_response(json_fenced, pkg)
    assert r2.verdict == VerificationVerdict.REJECTED_FALSE_ALARM
    assert r2.confidence == 0.88

    # 3. Invalid JSON fallback to UNCERTAIN
    bad_json = "I cannot determine if this is a crash or not."
    r3 = verifier._parse_vlm_response(bad_json, pkg)
    assert r3.verdict == VerificationVerdict.UNCERTAIN
    assert r3.is_uncertain is True


def test_qwen_verifier_connection_failure_fallback():
    # Point to an unroutable port to verify graceful fallback
    verifier = QwenMultimodalVerifier(
        api_base="http://127.0.0.1:59999/v1",
        timeout=0.5,
        mock_mode=False,
    )

    pkg = CandidateEvidencePackage(
        event_id="test_conn_fail",
        video_id="vid_03",
        candidate_frame=10,
        candidate_detector="single_vehicle",
        involved_track_ids=[1],
        keyframes={"trigger": np.zeros((10, 10, 3), dtype=np.uint8)},
        keyframe_indices={"trigger": 10},
    )

    res = verifier.verify(pkg)
    assert res.verdict == VerificationVerdict.UNCERTAIN
    assert "Connection error" in res.rationale or "API execution failure" in res.rationale or "Could not connect" in res.rationale


def test_universal_evidence_extractor_roi_and_som():
    extractor = CandidateEvidenceExtractor(roi_target_size=(128, 128), roi_expansion_factor=2.0)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Vehicle 1 at (200, 300, 60, 40)
    canvas[300:340, 200:260] = (255, 0, 0)
    # Vehicle 2 at (270, 310, 50, 40)
    canvas[310:350, 270:320] = (0, 255, 0)

    track_boxes = {
        1: (200.0, 300.0, 60.0, 40.0),
        2: (270.0, 310.0, 50.0, 40.0),
    }

    # 1. Verify ROI computation
    roi_bbox = extractor.compute_incident_roi((720, 1280), track_boxes, [1, 2])
    rx, ry, rw, rh = roi_bbox
    assert rw > 100
    assert rh > 80
    assert 0 <= rx <= 200
    assert 0 <= ry <= 300

    # 2. Verify evidence package with 5 temporal frames
    cached = {
        0: canvas,
        15: canvas,
        30: canvas,
        45: canvas,
        60: canvas,
    }
    pkg = extractor.extract_evidence_package(
        video_id="test_universal",
        candidate_frame=30,
        candidate_detector="pairwise_collision",
        involved_track_ids=[1, 2],
        telemetry={"tca_sec": 0.2, "dca_m": 0.4},
        track_boxes=track_boxes,
        cached_frames=cached,
    )

    assert set(pkg.full_frames.keys()) == {"t-2", "t-1", "t0", "t+1", "t+2"}
    assert set(pkg.roi_crops.keys()) == {"t-2", "t-1", "t0", "t+1", "t+2"}
    assert set(pkg.som_crops.keys()) == {"t-2", "t-1", "t0", "t+1", "t+2"}
    assert pkg.roi_crops["t0"].shape == (128, 128, 3)
    assert pkg.som_crops["t0"].shape == (128, 128, 3)


def test_qwen_verifier_ablation_ladder_levels():
    # Verify mock verification handles H0, H1, H2, H3, H4 correctly
    for level in ["H0", "H1", "H2", "H3", "H4"]:
        verifier = QwenMultimodalVerifier(mock_mode=True, ablation_level=level)
        assert verifier.ablation_level == level

        pkg = CandidateEvidencePackage(
            event_id=f"test_{level}",
            video_id="vid_test",
            candidate_frame=30,
            candidate_detector="pairwise_collision",
            involved_track_ids=[1, 2],
            keyframes={"trigger": np.zeros((50, 50, 3), dtype=np.uint8)},
            keyframe_indices={"trigger": 30},
            full_frames={"t0": np.zeros((50, 50, 3), dtype=np.uint8)},
            roi_crops={"t0": np.zeros((50, 50, 3), dtype=np.uint8)},
            som_crops={"t0": np.zeros((50, 50, 3), dtype=np.uint8)},
        )
        res = verifier.verify(pkg)
        assert res.is_confirmed is True

