"""Animal Hazard in Active Lane Candidate Detector."""

import logging
from typing import Dict, List, Set, Optional

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager

logger = logging.getLogger("road_surveillance.events.animal")


class AnimalHazardDetector(CandidateDetector):
    """Detects animal intrusion candidates in active roadway lanes, explicitly conditioned on detector capability."""

    DEFAULT_ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    }

    def __init__(
        self,
        min_consecutive_frames: int = 3,
        available_vocabulary: Optional[Dict[int, str]] = None,
    ):
        self.min_consecutive_frames = min_consecutive_frames
        self.intrusion_counts: Dict[int, int] = {}

        # Inspect detector vocabulary for animal capability
        self.supported_animal_labels: Set[str] = set()
        if available_vocabulary:
            vocab_lower = {v.lower() for v in available_vocabulary.values()}
            self.supported_animal_labels = vocab_lower.intersection(self.DEFAULT_ANIMAL_CLASSES)

        if not self.supported_animal_labels:
            logger.warning(
                "AnimalHazardDetector: Loaded detector vocabulary contains no recognizable animal classes. "
                "Animal hazard detection is capability-limited in this model checkpoint."
            )
        else:
            logger.info(
                f"AnimalHazardDetector initialized with {len(self.supported_animal_labels)} animal classes: "
                f"{sorted(self.supported_animal_labels)}"
            )

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
        candidates: List[EventCandidate] = []
        if not self.supported_animal_labels:
            return candidates  # Detector lacks animal class capability

        active_tids = set()
        for tid, kin in kinematics.items():
            tr = tracks.get(tid)
            if not tr or tr.class_label.lower() not in self.supported_animal_labels:
                continue

            active_tids.add(tid)
            if kin.is_in_active_lane:
                self.intrusion_counts[tid] = self.intrusion_counts.get(tid, 0) + 1
            else:
                self.intrusion_counts[tid] = 0

            if self.intrusion_counts[tid] >= self.min_consecutive_frames:
                candidates.append(
                    EventCandidate(
                        event_type="animal_hazard",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=min(0.9, 0.65 + 0.05 * self.intrusion_counts[tid]),
                        signals={
                            "animal_class": tr.class_label,
                            "consecutive_frames": self.intrusion_counts[tid],
                            "is_in_active_lane": kin.is_in_active_lane,
                            "capability_note": "Detector-capability dependent (pre-trained COCO vocabulary)",
                        },
                        ground_coordinates=kin.ground_pos,
                        lane_id=kin.lane_id,
                        description=f"Animal hazard candidate: Track {tid} ({tr.class_label}) in active lane {kin.lane_id or 'roadway'}",
                    )
                )

        # Cleanup
        for tid in list(self.intrusion_counts.keys()):
            if tid not in active_tids:
                del self.intrusion_counts[tid]

        return candidates
