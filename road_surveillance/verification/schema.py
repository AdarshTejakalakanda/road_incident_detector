"""Verification Schema Module for Phase 2 Multimodal Verification."""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


class VerificationVerdict(str, Enum):
    """Strict verification verdicts returned by the multimodal verifier."""
    CONFIRMED_INCIDENT = "CONFIRMED_INCIDENT"
    REJECTED_FALSE_ALARM = "REJECTED_FALSE_ALARM"
    UNCERTAIN = "UNCERTAIN"


class FinalIncidentDecision(str, Enum):
    """Explicit, unambiguous decision states produced by Asymmetric Fusion."""
    INCIDENT_CONFIRMED = "INCIDENT_CONFIRMED"
    INCIDENT_CONFIRMED_PENDING_REVIEW = "INCIDENT_CONFIRMED_PENDING_REVIEW"
    SUPPRESSED_FALSE_ALARM = "SUPPRESSED_FALSE_ALARM"
    FLAGGED_FOR_HUMAN_REVIEW = "FLAGGED_FOR_HUMAN_REVIEW"
    NORMAL_FLOW = "NORMAL_FLOW"


@dataclass
class IncidentVerificationResult:
    """Structured incident record output produced by the multimodal verifier."""
    event_id: str
    video_id: str
    candidate_frame: int
    verdict: VerificationVerdict
    confidence: float
    rationale: str
    detected_crash_type: str
    involved_tracks: List[int] = field(default_factory=list)
    raw_response: Optional[str] = None
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.verdict == VerificationVerdict.CONFIRMED_INCIDENT

    @property
    def is_rejected(self) -> bool:
        return self.verdict == VerificationVerdict.REJECTED_FALSE_ALARM

    @property
    def is_uncertain(self) -> bool:
        return self.verdict == VerificationVerdict.UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentVerificationResult":
        verdict = VerificationVerdict(data["verdict"])
        return cls(
            event_id=data["event_id"],
            video_id=data["video_id"],
            candidate_frame=int(data["candidate_frame"]),
            verdict=verdict,
            confidence=float(data.get("confidence", 0.0)),
            rationale=str(data.get("rationale", "")),
            detected_crash_type=str(data.get("detected_crash_type", "unknown")),
            involved_tracks=list(data.get("involved_tracks", [])),
            raw_response=data.get("raw_response"),
            telemetry_summary=dict(data.get("telemetry_summary", {})),
        )
