"""Event Candidate Definitions and Base Class."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..tracking.track_state import Track, TrackletState
from ..geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ..calibration.lane_manager import LaneManager


@dataclass
class SpatialPoint:
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: Any) -> float:
        if hasattr(other, "x") and hasattr(other, "y"):
            ox, oy = other.x, other.y
        elif isinstance(other, (tuple, list)) and len(other) >= 2:
            ox, oy = other[0], other[1]
        else:
            return 0.0
        return float(((self.x - ox) ** 2 + (self.y - oy) ** 2) ** 0.5)


@dataclass
class EventCandidate:
    """Standardized Road Surveillance Event Candidate."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "unknown"
    video_id: str = ""
    frame_idx: int = 0
    timestamp_ns: int = 0
    track_ids: List[int] = field(default_factory=list)
    primary_track_id: int = 0
    confidence: float = 0.0
    signals: Dict[str, Any] = field(default_factory=dict)
    ground_coordinates: Tuple[float, float] = (0.0, 0.0)
    lane_id: Optional[str] = None
    is_candidate: bool = True  # Explicit candidate marker per specification
    description: str = ""

    @property
    def participating_track_ids(self) -> List[int]:
        return self.track_ids

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_ns * 1e-9 if self.timestamp_ns > 0 else self.frame_idx / 30.0

    @property
    def speed_kmh(self) -> float:
        signals = self.signals or {}
        return float(signals.get("metric_speed_kmh", signals.get("curr_spd", signals.get("speed_kmh", signals.get("speed_1_kmh", 0.0)))))

    @property
    def spatial_centroid(self) -> SpatialPoint:
        return SpatialPoint(self.ground_coordinates[0], self.ground_coordinates[1])


class CandidateDetector(ABC):
    """Abstract base class for all incident candidate detectors."""

    @abstractmethod
    def evaluate(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
        lane_manager: LaneManager,
    ) -> List[EventCandidate]:
        """Evaluate tracks and kinematics for candidate event triggers."""
        pass
