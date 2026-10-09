"""Debris Candidate Detector."""

import logging
from typing import Dict, List, Set, Optional

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager

logger = logging.getLogger("road_surveillance.events.debris")


class DebrisCandidateDetector(CandidateDetector):
    """Detects potential stationary obstacle / debris candidates on active carriageways.
    Explicitly documented as dependent on detector capability.
    """

    KNOWN_DEBRIS_CLASSES = {
        "backpack", "suitcase", "sports ball", "box", "package", "bottle"
    }

    NON_DEBRIS_CLASSES = {
        "person", "car", "motorcycle", "bus", "truck", "bicycle", "traffic light", "stop sign"
    }

    def __init__(
        self,
        min_stationary_sec: float = 4.0,
        available_vocabulary: Optional[Dict[int, str]] = None,
    ):
        self.min_stationary_sec = min_stationary_sec
        self.debris_vocabulary: Set[str] = set()

        if available_vocabulary:
            vocab_lower = {v.lower() for v in available_vocabulary.values()}
            self.debris_vocabulary = vocab_lower.intersection(self.KNOWN_DEBRIS_CLASSES)

        logger.info(
            f"DebrisCandidateDetector initialized. Recognized obstacle classes in detector: "
            f"{sorted(self.debris_vocabulary) if self.debris_vocabulary else 'None (limited to static non-vehicle tracklets)'}"
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

        for tid, kin in kinematics.items():
            tr = tracks.get(tid)
            if not tr:
                continue

            c_label = tr.class_label.lower()
            if c_label in self.NON_DEBRIS_CLASSES:
                continue

            # Only flag as debris if the class is in the known-obstacle allowlist.
            # Reject any class that is NOT explicitly recognised as a road obstacle
            # (e.g. 'clock', 'toothbrush', 'teddy bear' must NOT trigger debris alerts).
            is_potential_debris = c_label in self.debris_vocabulary

            if is_potential_debris and kin.is_in_active_lane and kin.stationary_duration_sec >= self.min_stationary_sec:
                candidates.append(
                    EventCandidate(
                        event_type="debris_candidate",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=0.55 if c_label in self.debris_vocabulary else 0.40,
                        signals={
                            "detected_label": tr.class_label,
                            "stationary_duration_sec": kin.stationary_duration_sec,
                            "bbox_xywh": tr.current_bbox_xywh,
                            "ground_coordinates": kin.ground_pos,
                            "is_in_active_lane": kin.is_in_active_lane,
                            "detector_capability_note": "Stock YOLO26 COCO weights cannot reliably detect generic road debris. Marked as candidate only.",
                        },
                        ground_coordinates=kin.ground_pos,
                        lane_id=kin.lane_id,
                        description=(
                            f"Debris candidate: Track {tid} ({tr.class_label}) stationary for "
                            f"{kin.stationary_duration_sec:.1f}s in active lane {kin.lane_id or 'roadway'}"
                        ),
                    )
                )

        return candidates
