"""Unit tests for Two-Tier Incident Arbiter."""

import pytest
import numpy as np

from road_surveillance.verification.two_tier_arbiter import (
    TwoTierIncidentArbiter,
    DispatchRouting,
    ArbitratedIncidentDecision,
)
from road_surveillance.verification.schema import VerificationVerdict, IncidentVerificationResult
from road_surveillance.verification.base import CandidateEvidencePackage, IncidentVerifier
from road_surveillance.events.base import EventCandidate
from road_surveillance.evidence.consolidator import IncidentEpisode


class MockVLMVerifier(IncidentVerifier):
    """Mock verifier for testing two-tier routing."""
    def __init__(self, verdict=VerificationVerdict.CONFIRMED_INCIDENT, confidence=0.88):
        self.verdict = verdict
        self.confidence = confidence
        self.invocations = 0

    def verify(self, evidence: CandidateEvidencePackage) -> IncidentVerificationResult:
        self.invocations += 1
        return IncidentVerificationResult(
            event_id=evidence.event_id,
            video_id=evidence.video_id,
            candidate_frame=evidence.candidate_frame,
            verdict=self.verdict,
            confidence=self.confidence,
            rationale="Mock VLM verification decision",
            detected_crash_type="rear_end" if self.verdict == VerificationVerdict.CONFIRMED_INCIDENT else "none",
        )


def make_dummy_evidence_package(ep_id="ep_01", vid_id="vid_01", f=50):
    return CandidateEvidencePackage(
        event_id=ep_id,
        video_id=vid_id,
        candidate_frame=f,
        candidate_detector="collision",
        involved_track_ids=[1, 2],
        keyframes={"pre": np.zeros((10, 10, 3)), "trigger": np.zeros((10, 10, 3)), "post": np.zeros((10, 10, 3))},
        keyframe_indices={"pre": f - 15, "trigger": f, "post": f + 15},
    )


def test_high_confidence_collision_fast_paths_direct_alert():
    mock_vlm = MockVLMVerifier()
    arbiter = TwoTierIncidentArbiter(vlm_verifier=mock_vlm)

    # Episode with 3 severe collision triggers
    triggers = [
        EventCandidate(
            event_type="collision",
            frame_idx=50 + i,
            track_ids=[1, 2],
            signals={"bbox_iou": 0.25, "accel_1_ms2": -7.5, "accel_2_ms2": -8.0},
        )
        for i in range(3)
    ]
    episode = IncidentEpisode(
        episode_id="ep_01",
        video_id="test_vid",
        start_frame=50,
        last_trigger_frame=52,
        primary_detector="collision",
        involved_track_ids={1, 2},
        raw_triggers=triggers,
    )

    decision = arbiter.arbitrate(episode, make_dummy_evidence_package())

    assert decision.routing == DispatchRouting.DIRECT_ALERT
    assert decision.verdict == VerificationVerdict.CONFIRMED_INCIDENT
    assert decision.vlm_invoked is False
    assert mock_vlm.invocations == 0
    assert arbiter.direct_alerts == 1
    assert arbiter.vlm_invocations == 0
    assert arbiter.vlm_invocation_rate == 0.0


def test_high_confidence_single_vehicle_rollover_direct_alert():
    mock_vlm = MockVLMVerifier()
    arbiter = TwoTierIncidentArbiter(vlm_verifier=mock_vlm)

    triggers = [
        EventCandidate(
            event_type="single_vehicle",
            frame_idx=40,
            track_ids=[5],
            signals={"anomaly_type": "rollover", "aspect_ratio_change": 0.55, "yaw_rate_deg_s": 42.0},
        )
    ]
    episode = IncidentEpisode(
        episode_id="ep_rollover",
        video_id="test_vid",
        start_frame=40,
        last_trigger_frame=40,
        primary_detector="single_vehicle",
        involved_track_ids={5},
        raw_triggers=triggers,
    )

    decision = arbiter.arbitrate(episode, make_dummy_evidence_package(f=40))

    assert decision.routing == DispatchRouting.DIRECT_ALERT
    assert decision.verdict == VerificationVerdict.CONFIRMED_INCIDENT
    assert decision.detected_crash_type == "rollover"
    assert decision.vlm_invoked is False
    assert mock_vlm.invocations == 0


def test_ambiguous_candidate_routes_to_vlm():
    mock_vlm = MockVLMVerifier(verdict=VerificationVerdict.REJECTED_FALSE_ALARM)
    arbiter = TwoTierIncidentArbiter(vlm_verifier=mock_vlm)

    # Distant conflict near-miss: moderate decel (-3.8), no bounding box IoU
    triggers = [
        EventCandidate(
            event_type="distant_conflict",
            frame_idx=100,
            track_ids=[10, 11],
            signals={"distance_m": 8.5, "tca_sec": 1.2, "metric_accel_1": -3.8},
        )
    ]
    episode = IncidentEpisode(
        episode_id="ep_distant",
        video_id="test_vid",
        start_frame=100,
        last_trigger_frame=100,
        primary_detector="distant_conflict",
        involved_track_ids={10, 11},
        raw_triggers=triggers,
    )

    decision = arbiter.arbitrate(episode, make_dummy_evidence_package(f=100))

    assert decision.routing == DispatchRouting.DISCARDED
    assert decision.verdict == VerificationVerdict.REJECTED_FALSE_ALARM
    assert decision.vlm_invoked is True
    assert mock_vlm.invocations == 1
    assert arbiter.vlm_rejected == 1
    assert arbiter.vlm_invocation_rate == 1.0


def test_vlm_invocation_rate_calculation():
    mock_vlm = MockVLMVerifier(verdict=VerificationVerdict.CONFIRMED_INCIDENT)
    arbiter = TwoTierIncidentArbiter(vlm_verifier=mock_vlm)

    # 4 high confidence + 1 ambiguous candidate = 5 total candidates
    # VLM invocation rate should be 1/5 = 20% (<= 20% target)
    for i in range(4):
        triggers = [
            EventCandidate(
                event_type="collision",
                frame_idx=20 * i,
                track_ids=[1, 2],
                signals={"bbox_iou": 0.30, "accel_1_ms2": -8.0, "accel_2_ms2": -7.0},
            )
            for _ in range(3)
        ]
        ep = IncidentEpisode(
            episode_id=f"direct_{i}",
            video_id="vid",
            start_frame=20 * i,
            last_trigger_frame=20 * i + 2,
            primary_detector="collision",
            raw_triggers=triggers,
        )
        arbiter.arbitrate(ep, make_dummy_evidence_package())

    # 1 ambiguous candidate
    amb_triggers = [
        EventCandidate(
            event_type="distant_conflict",
            frame_idx=90,
            track_ids=[3, 4],
            signals={"distance_m": 5.0, "metric_accel_1": -4.0},
        )
    ]
    amb_ep = IncidentEpisode(
        episode_id="amb_1",
        video_id="vid",
        start_frame=90,
        last_trigger_frame=90,
        primary_detector="distant_conflict",
        raw_triggers=amb_triggers,
    )
    arbiter.arbitrate(amb_ep, make_dummy_evidence_package())

    assert arbiter.total_candidates == 5
    assert arbiter.direct_alerts == 4
    assert arbiter.vlm_invocations == 1
    assert arbiter.vlm_invocation_rate == pytest.approx(0.20)
