"""Wrong-Way Vehicle Incident Candidate Detector."""

from typing import Dict, List
import numpy as np

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence, calculate_directional_cosine
from ...calibration.lane_manager import LaneManager


class WrongWayDetector(CandidateDetector):
    """Detects vehicles traveling contrary to permitted lane direction in defined lanes."""

    VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "van", "train"}

    def __init__(
        self,
        min_speed_kmh: float = 5.0,
        cos_threshold: float = -0.5,
        min_consecutive_frames: int = 4,
    ):
        self.min_speed_kmh = min_speed_kmh
        self.cos_threshold = cos_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.violation_counts: Dict[int, int] = {}

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
        active_tids = set()

        for tid, kin in kinematics.items():
            track = tracks.get(tid)
            if not track or track.class_label.lower() not in self.VEHICLE_LABELS:
                continue

            active_tids.add(tid)

            # Must be in a lane with defined direction and moving fast enough
            lane = lane_manager.get_lane_for_ground_point(kin.ground_pos[0], kin.ground_pos[1])
            if not lane or kin.speed_kmh < self.min_speed_kmh:
                self.violation_counts[tid] = 0
                continue

            cos_theta = calculate_directional_cosine(kin.heading_vector, lane.permitted_direction)

            if cos_theta < self.cos_threshold:
                self.violation_counts[tid] = self.violation_counts.get(tid, 0) + 1
            else:
                self.violation_counts[tid] = 0

            if self.violation_counts[tid] >= self.min_consecutive_frames:
                candidates.append(
                    EventCandidate(
                        event_type="wrong_way_vehicle",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=min(1.0, 0.7 + (abs(cos_theta) * 0.3)),
                        signals={
                            "directional_cosine": cos_theta,
                            "cos_threshold": self.cos_threshold,
                            "speed_kmh": kin.speed_kmh,
                            "consecutive_frames": self.violation_counts[tid],
                            "lane_permitted_direction": lane.permitted_direction,
                            "vehicle_heading_vector": kin.heading_vector,
                        },
                        ground_coordinates=kin.ground_pos,
                        lane_id=lane.lane_id,
                        description=(
                            f"Track {tid} ({track.class_label}) traveling wrong-way in lane "
                            f"{lane.lane_id} at {kin.speed_kmh:.1f} km/h (cos={cos_theta:.2f})"
                        ),
                    )
                )

        # Cleanup
        for tid in list(self.violation_counts.keys()):
            if tid not in active_tids:
                del self.violation_counts[tid]

        return candidates

    def reset(self) -> None:
        self.violation_counts.clear()
