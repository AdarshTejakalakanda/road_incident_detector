"""Two-Tier Selective Incident Arbitration Engine.

Architecture:
Tier 1: High-Confidence Deterministic Fast-Path
  - Unequivocal physical collisions (severe decel + high IoU / close contact)
  - Severe single-vehicle rollovers / catastrophic barrier impacts
  - Sustained wrong-way high-speed conflicts
  -> Dispatches immediate ITS alert without VLM latency.

Tier 2: Multimodal VLM Selective Verification (Qwen3.8-27B)
  - Distant trajectory conflicts (near-miss vs glancing collision)
  - Borderline / moderate decelerations without clear geometric overlap
  - Single-vehicle decelerations without structural deformation or rollover
  - Occluded or low-maturity track intersections
  -> Routes 5 strategic keyframes + kinematic telemetry to Qwen3.8-27B.
  -> Filters out deterministic false alarms while keeping VLM invocation rate <= 20%.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .schema import VerificationVerdict, IncidentVerificationResult, FinalIncidentDecision
from .base import CandidateEvidencePackage, IncidentVerifier
from ..evidence.consolidator import IncidentEpisode

logger = logging.getLogger("road_surveillance.verification.two_tier_arbiter")


class DispatchRouting(str, Enum):
    """Routing path taken by the arbiter."""
    DIRECT_ALERT = "DIRECT_ALERT"          # Fast-path deterministic dispatch (0s VLM latency)
    VLM_VERIFIED = "VLM_VERIFIED"          # Routed to Multimodal VLM for deep reasoning
    DISCARDED = "DISCARDED"                # Filtered out prior to dispatch / rejected


@dataclass
class ArbitratedIncidentDecision:
    """Final decision produced by the Two-Tier Arbitration Engine."""
    event_id: str
    video_id: str
    trigger_frame: int
    routing: DispatchRouting
    verdict: VerificationVerdict
    confidence: float
    rationale: str
    detected_crash_type: str
    vlm_invoked: bool
    final_decision: FinalIncidentDecision = FinalIncidentDecision.INCIDENT_CONFIRMED
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    raw_vlm_result: Optional[IncidentVerificationResult] = None

    @property
    def should_alert(self) -> bool:
        return self.final_decision in (
            FinalIncidentDecision.INCIDENT_CONFIRMED,
            FinalIncidentDecision.INCIDENT_CONFIRMED_PENDING_REVIEW,
        )


class TwoTierIncidentArbiter:
    """Two-tier incident arbiter balancing real-time speed and multimodal precision."""

    def __init__(
        self,
        vlm_verifier: Optional[IncidentVerifier] = None,
        severe_decel_ms2: float = -6.0,
        severe_single_decel_ms2: float = -12.0,
        high_iou_threshold: float = 0.15,
        min_direct_persistence_frames: int = 3,
    ):
        self.vlm_verifier = vlm_verifier
        self.severe_decel_ms2 = severe_decel_ms2
        self.severe_single_decel_ms2 = severe_single_decel_ms2
        self.high_iou_threshold = high_iou_threshold
        self.min_direct_persistence_frames = min_direct_persistence_frames

        # Telemetry metrics
        self.total_candidates = 0
        self.direct_alerts = 0
        self.vlm_invocations = 0
        self.vlm_confirmed = 0
        self.vlm_rejected = 0
        self.vlm_uncertain = 0

    def reset_stats(self):
        self.total_candidates = 0
        self.direct_alerts = 0
        self.vlm_invocations = 0
        self.vlm_confirmed = 0
        self.vlm_rejected = 0
        self.vlm_uncertain = 0

    @property
    def vlm_invocation_rate(self) -> float:
        if self.total_candidates == 0:
            return 0.0
        return self.vlm_invocations / self.total_candidates

    def is_deterministic_high_confidence(self, episode: IncidentEpisode) -> Tuple[bool, str, str]:
        """Evaluate if an episode qualifies for high-confidence direct dispatch.
        
        Returns:
            (is_high_conf, crash_type, rationale)
        """
        detector = episode.primary_detector.lower()
        triggers = episode.raw_triggers
        n_triggers = len(triggers)

        # 1. Pairwise Collision Fast-Path
        if "collision" in detector or "pairwise" in detector:
            max_iou = 0.0
            min_accel = 0.0
            for t in triggers:
                sig = t.signals or {}
                iou = float(sig.get("bbox_iou", sig.get("iou", 0.0)))
                max_iou = max(max_iou, iou)
                a1 = float(sig.get("accel_1_ms2", sig.get("accel_1", 0.0)))
                a2 = float(sig.get("accel_2_ms2", sig.get("accel_2", 0.0)))
                min_accel = min(min_accel, a1, a2)

            # High confidence: direct bounding box intersection + severe impact deceleration
            if max_iou >= self.high_iou_threshold and min_accel <= self.severe_decel_ms2 and n_triggers >= self.min_direct_persistence_frames:
                return True, "pairwise_collision", f"High-confidence collision: IoU {max_iou:.2f} >= {self.high_iou_threshold} and decel {min_accel:.1f} m/s^2 <= {self.severe_decel_ms2} m/s^2 across {n_triggers} frames"

        # 2. Single-Vehicle Anomaly Fast-Path
        if "single" in detector:
            for t in triggers:
                sig = t.signals or {}
                atype = str(sig.get("anomaly_type", ""))
                accel = float(sig.get("metric_accel_ms2", sig.get("accel_1_ms2", 0.0)))
                ar_change = float(sig.get("aspect_ratio_change", 0.0))
                yaw_rate = float(sig.get("yaw_rate_deg_s", 0.0))

                # Severe rollover or violent barrier strike
                if atype == "rollover" or (ar_change >= 0.40 and abs(yaw_rate) >= 30.0):
                    return True, "rollover", f"High-confidence rollover: aspect ratio change {ar_change:.2f} with yaw rate {yaw_rate:.1f} deg/s"
                if atype == "severe_barrier_impact" or (accel <= self.severe_single_decel_ms2 and n_triggers >= 3):
                    return True, "single_vehicle_barrier", f"High-confidence barrier impact: severe deceleration {accel:.1f} m/s^2 across {n_triggers} triggers"

        # 3. Wrong-Way Fast-Path
        if "wrong_way" in detector:
            if n_triggers >= 6:
                return True, "wrong_way", f"Persistent high-confidence wrong-way travel sustained for {n_triggers} consecutive frames"

        return False, "unknown", ""

    def arbitrate(
        self,
        episode: IncidentEpisode,
        evidence_package: Optional[CandidateEvidencePackage] = None,
    ) -> ArbitratedIncidentDecision:
        """Arbitrate an incident candidate episode through the Two-Tier routing pipeline."""
        self.total_candidates += 1
        ev_id = episode.episode_id
        vid_id = episode.video_id
        trigger_f = episode.peak_impact_frame or episode.start_frame

        # Step 1: Check Deterministic High-Confidence Fast-Path
        is_fast_path, crash_type, fast_rationale = self.is_deterministic_high_confidence(episode)
        if is_fast_path:
            self.direct_alerts += 1
            logger.info(f"[{vid_id}] Episode {ev_id} fast-pathed as DIRECT_ALERT: {fast_rationale}")
            return ArbitratedIncidentDecision(
                event_id=ev_id,
                video_id=vid_id,
                trigger_frame=trigger_f,
                routing=DispatchRouting.DIRECT_ALERT,
                verdict=VerificationVerdict.CONFIRMED_INCIDENT,
                confidence=0.95,
                rationale=fast_rationale,
                detected_crash_type=crash_type,
                vlm_invoked=False,
                telemetry_summary={
                    "primary_detector": episode.primary_detector,
                    "triggers_count": len(episode.raw_triggers),
                    "peak_metric_score": episode.peak_metric_score,
                },
            )

        # Step 2: Ambiguous / Borderline Candidate -> Route to Multimodal VLM
        if self.vlm_verifier is None or evidence_package is None:
            # Fallback when VLM is unconfigured: preserve candidate conservatively
            logger.warning(f"[{vid_id}] VLM not available for borderline episode {ev_id}. Defaulting to conservative alert.")
            return ArbitratedIncidentDecision(
                event_id=ev_id,
                video_id=vid_id,
                trigger_frame=trigger_f,
                routing=DispatchRouting.DIRECT_ALERT,
                verdict=VerificationVerdict.CONFIRMED_INCIDENT,
                confidence=0.60,
                rationale=f"Borderline trigger from {episode.primary_detector} (VLM verification bypassed)",
                detected_crash_type="borderline_unverified",
                vlm_invoked=False,
                telemetry_summary={"primary_detector": episode.primary_detector},
            )

        self.vlm_invocations += 1
        logger.info(f"[{vid_id}] Routing ambiguous episode {ev_id} ({episode.primary_detector}) to Qwen3.8-27B...")
        vlm_res = self.vlm_verifier.verify(evidence_package)

        if vlm_res.verdict == VerificationVerdict.CONFIRMED_INCIDENT:
            self.vlm_confirmed += 1
            routing = DispatchRouting.VLM_VERIFIED
            final_decision = FinalIncidentDecision.INCIDENT_CONFIRMED
        elif vlm_res.verdict == VerificationVerdict.REJECTED_FALSE_ALARM:
            self.vlm_rejected += 1
            routing = DispatchRouting.DISCARDED
            final_decision = FinalIncidentDecision.SUPPRESSED_FALSE_ALARM
        else:
            self.vlm_uncertain += 1
            routing = DispatchRouting.VLM_VERIFIED
            final_decision = FinalIncidentDecision.FLAGGED_FOR_HUMAN_REVIEW

        return ArbitratedIncidentDecision(
            event_id=ev_id,
            video_id=vid_id,
            trigger_frame=trigger_f,
            routing=routing,
            verdict=vlm_res.verdict,
            confidence=vlm_res.confidence,
            rationale=vlm_res.rationale,
            detected_crash_type=vlm_res.detected_crash_type,
            vlm_invoked=True,
            final_decision=final_decision,
            telemetry_summary=vlm_res.telemetry_summary,
            raw_vlm_result=vlm_res,
        )
