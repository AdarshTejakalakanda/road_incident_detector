"""Abstract Base Verifier and Evidence Package Data Structure."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .schema import IncidentVerificationResult


@dataclass
class CandidateEvidencePackage:
    """Encapsulates the multimodal evidence for an incident candidate episode."""
    event_id: str
    video_id: str
    candidate_frame: int
    candidate_detector: str
    involved_track_ids: List[int]
    keyframes: Dict[str, np.ndarray]  # {"pre": image, "trigger": image, "post": image}
    keyframe_indices: Dict[str, int]  # {"pre": f-15, "trigger": f, "post": f+15}
    full_frames: Dict[str, np.ndarray] = field(default_factory=dict)  # {"t-2": img, ...}
    roi_crops: Dict[str, np.ndarray] = field(default_factory=dict)  # {"t-2": crop, ...}
    som_crops: Dict[str, np.ndarray] = field(default_factory=dict)  # {"t-2": marked_crop, ...}
    roi_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) in original canvas
    temporal_indices: Dict[str, int] = field(default_factory=dict)  # {"t-2": idx, ...}
    telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IncidentVerifier(ABC):
    """Abstract base class for downstream incident verifiers."""

    @abstractmethod
    def verify(self, evidence: CandidateEvidencePackage) -> IncidentVerificationResult:
        """Verify a single candidate evidence package.

        Args:
            evidence: CandidateEvidencePackage containing keyframes, telemetry, and metadata.

        Returns:
            IncidentVerificationResult with verdict, confidence, rationale, and crash type.
        """
        pass

    def batch_verify(self, evidence_list: List[CandidateEvidencePackage]) -> List[IncidentVerificationResult]:
        """Verify a batch of candidate evidence packages."""
        return [self.verify(ev) for ev in evidence_list]
