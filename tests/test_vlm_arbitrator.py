"""Unit tests for VLM Arbitration, Filmstrip Builder, and Groq Arbitrator."""

import numpy as np
import pytest

from modules.vlm_arbitrator import (
    create_evidence_strip,
    passes_telemetry_prefilter,
    GroqArbitrator,
)


def test_create_evidence_strip():
    crop1 = np.ones((100, 150, 3), dtype=np.uint8) * 50
    crop2 = np.ones((120, 160, 3), dtype=np.uint8) * 100
    crop3 = np.ones((80, 120, 3), dtype=np.uint8) * 150

    strip = create_evidence_strip(crop1, crop2, crop3, target_h=336)
    assert strip.shape[0] == 336
    assert strip.shape[2] == 3
    # Total width should be sum of 3 resized widths + two 3px dividers
    assert strip.shape[1] > 336

    # Test with empty/None crops
    strip_empty = create_evidence_strip(np.empty((0, 0, 3)), None, crop3, target_h=336)
    assert strip_empty.shape[0] == 336
    assert strip_empty.shape[2] == 3


def test_passes_telemetry_prefilter():
    # Pass case: severe deceleration and proximity
    tel_pass = {
        "accel_ms2": -5.2,
        "distance_m": 0.6,
        "track_age": 12,
        "bbox_iou": 0.25,
    }
    passes, msg = passes_telemetry_prefilter(tel_pass)
    assert passes is True

    # Pass case: positive impulse / jump in longitudinal acceleration
    tel_impulse = {
        "accel_ms2": 11.2,
        "distance_m": 0.4,
        "track_age": 15,
        "bbox_iou": 0.30,
    }
    passes, msg = passes_telemetry_prefilter(tel_impulse)
    assert passes is True

    # Fail case: young track
    tel_young = {
        "accel_ms2": -6.0,
        "distance_m": 0.5,
        "track_age": 2,
    }
    passes, msg = passes_telemetry_prefilter(tel_young)
    assert passes is False
    assert "young" in msg.lower()

    # Fail case: normal flow (no deceleration, far distance)
    tel_normal = {
        "accel_ms2": -0.5,
        "distance_m": 12.0,
        "track_age": 25,
        "bbox_iou": 0.0,
    }
    passes, msg = passes_telemetry_prefilter(tel_normal)
    assert passes is False


def test_groq_arbitrator_fallback_and_caching():
    arbiter = GroqArbitrator(api_key=None, mock_fallback=True)

    dummy_strip = np.zeros((336, 1000, 3), dtype=np.uint8)
    # Test 1: Ordinary kinematic candidate without bypass-level contact/impulse -> CANDIDATE_UNCONFIRMED
    telemetry_unconfirmed = {
        "event_type": "collision",
        "involved_tracks": [1, 2],
        "accel_ms2": 11.2,
        "distance_m": 0.5,
        "bbox_iou": 0.22,
        "track_age": 14,
    }

    res_unconfirmed = arbiter.arbitrate(dummy_strip, telemetry_unconfirmed, candidate_frame=51, video_id="test_vid")
    assert res_unconfirmed["verdict"] == "CANDIDATE_UNCONFIRMED"
    assert res_unconfirmed["is_collision"] is False
    assert res_unconfirmed["suppressed"] is True
    assert res_unconfirmed["source"] == "offline_unverified_fallback"

    # Test 2: Severe impact meeting deterministic high-confidence bypass -> CONFIRMED_INCIDENT
    telemetry_bypass = {
        "event_type": "collision",
        "involved_tracks": [1, 2],
        "accel_ms2": -18.5,
        "distance_m": 0.5,
        "bbox_iou": 0.38,
        "delta_v_drop": 20.0,
        "tau_rest": 0.40,
        "track_age": 14,
    }

    result = arbiter.arbitrate(dummy_strip, telemetry_bypass, candidate_frame=52, video_id="test_vid")
    assert result["verdict"] == "CONFIRMED_INCIDENT"
    assert result["is_collision"] is True
    assert result["collision_type"] == "rear_end_collision"
    assert result["confidence"] > 0.80
    assert result["source"] == "deterministic_high_confidence_bypass"
    assert len(result["visual_cues"]) > 0

    # Test cache retrieval
    cached = arbiter.arbitrate(dummy_strip, telemetry_bypass, candidate_frame=52, video_id="test_vid")
    assert cached == result



# Stage T Math-Validated Mandatory Unit Tests
from modules.vlm_arbitrator import passes_tier1_admission, validate_stage_t_vlm_output
from modules.evidence_builder import select_temporal_evidence_frames

def test_stage_t_normal_braking_rejection():
    tel_braking = {
        'event_type': 'collision',
        'accel_ms2': -3.8,
        'delta_v_drop': 16.0,
        'norm_pixel_dist': 1.10,
        'homography_calibrated': False,
        'bbox_iou': 0.0,
        'tau_rest': 0.50,
    }
    passes, reason = passes_tier1_admission(tel_braking)
    assert passes is False
    assert 'a_long' in reason or 'spatial_convergence' in reason

def test_stage_t_outer_lane_cruising_rejection():
    tel_cruising = {
        'event_type': 'single_vehicle_anomaly',
        'd_edge_m': 1.2,
        'accel_ms2': -2.0,
        'v_dot_n': 0.05,
    }
    passes, reason = passes_tier1_admission(tel_cruising)
    assert passes is False
    assert 'rejected' in reason.lower()

def test_stage_t_no_deformation_means_no_vlm_confirmation():
    arbiter = GroqArbitrator(api_key=None, mock_fallback=False)
    dummy_strip = np.zeros((336, 1008, 3), dtype=np.uint8)
    tel = {
        'event_type': 'collision',
        'accel_ms2': -8.0,
        'delta_v_drop': 15.0,
        'tau_rest': 0.50,
        'ground_distance_m': 1.5,
        'homography_calibrated': True,
        'bbox_iou': 0.15,
    }
    mock_vlm = {
        'physical_contact_observed': True,
        'structural_damage_or_deformation': False,
        'post_impact_displacement_or_rest': True,
        'benign_alternative': 'none',
        'confidence': 0.85,
        'collision_type': 'rear_end_collision',
        'rationale': 'Vehicles touched but no bumper deformation visible',
    }
    result = arbiter.arbitrate(dummy_strip, tel, candidate_frame=100, video_id='test_vid', mock_vlm_response=mock_vlm)
    assert result['verdict'] == 'CANDIDATE_UNCONFIRMED'
    assert result['is_collision'] is False
    assert result['suppressed'] is True

def test_stage_t_vlm_cannot_override_deterministic_bypass():
    arbiter = GroqArbitrator(api_key=None, mock_fallback=False)
    dummy_strip = np.zeros((336, 1008, 3), dtype=np.uint8)
    tel_high_conf = {
        'event_type': 'collision',
        'accel_ms2': -14.0,
        'bbox_iou': 0.35,
        'tau_rest': 0.40,
        'speed_kmh': 0.0,
        'max_recent_speed_kmh': 45.0,
    }
    mock_vlm_negative = {
        'physical_contact_observed': False,
        'structural_damage_or_deformation': False,
        'post_impact_displacement_or_rest': False,
        'benign_alternative': 'normal_turn',
        'confidence': 0.90,
        'collision_type': 'none',
        'rationale': 'VLM mistakenly claims normal turn',
    }
    result = arbiter.arbitrate(dummy_strip, tel_high_conf, candidate_frame=200, video_id='test_vid', mock_vlm_response=mock_vlm_negative)
    assert result['verdict'] == 'CONFIRMED_INCIDENT'
    assert result['is_collision'] is True
    assert result['source'] == 'deterministic_high_confidence_bypass'

def test_stage_t_pts_evidence_timing():
    f_pre_118, f_impact_118, f_post_118 = select_temporal_evidence_frames(candidate_frame=118, candidate_timestamp_s=10.0, fps=11.8, total_frames=300)
    assert f_impact_118 == 118
    # dt_pre = 1.0s * 11.8 fps = round(11.8) = 12 frames
    assert f_pre_118 == 118 - 12
    # dt_post = 1.5s * 11.8 fps = round(17.7) = 18 frames
    assert f_post_118 == 118 + 18

    f_pre_30, f_impact_30, f_post_30 = select_temporal_evidence_frames(candidate_frame=300, candidate_timestamp_s=10.0, fps=30.0, total_frames=600)
    assert f_impact_30 == 300
    # dt_pre = 1.0s * 30 fps = 30 frames
    assert f_pre_30 == 300 - 30
    # dt_post = 1.5s * 30 fps = 45 frames
    assert f_post_30 == 300 + 45

    # Test with explicit frame_pts_map
    pts_map = {f: f / 30.0 for f in range(1, 600)}
    f_pre_pts, f_imp_pts, f_post_pts = select_temporal_evidence_frames(candidate_frame=300, candidate_timestamp_s=10.0, fps=30.0, frame_pts_map=pts_map)
    assert f_imp_pts == 300
    assert abs(pts_map[f_pre_pts] - 9.0) < 0.05
    assert abs(pts_map[f_post_pts] - 11.5) < 0.05
