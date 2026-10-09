"""Pedestrian Hazard in Active Roadway Candidate Detector."""

from typing import Dict, List

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager


class PedestrianHazardDetector(CandidateDetector):
    """Detects pedestrian intrusion candidates inside active road lanes."""

    def __init__(self, min_consecutive_frames: int = 3):
        self.min_consecutive_frames = min_consecutive_frames
        self.intrusion_counts: Dict[int, int] = {}

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
            tr = tracks.get(tid)
            if not tr or tr.class_label.lower() != "person":
                continue

            active_tids.add(tid)

            if kin.is_in_active_lane:
                self.intrusion_counts[tid] = self.intrusion_counts.get(tid, 0) + 1
            else:
                self.intrusion_counts[tid] = 0

            if self.intrusion_counts[tid] >= self.min_consecutive_frames:
                candidates.append(
                    EventCandidate(
                        event_type="pedestrian_hazard",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=min(0.95, 0.7 + 0.05 * self.intrusion_counts[tid]),
                        signals={
                            "consecutive_frames_in_lane": self.intrusion_counts[tid],
                            "ground_coordinates": kin.ground_pos,
                            "speed_kmh": kin.speed_kmh,
                            "is_in_active_lane": kin.is_in_active_lane,
                        },
                        ground_coordinates=kin.ground_pos,
                        lane_id=kin.lane_id,
                        description=(
                            f"Pedestrian hazard candidate: Track {tid} in active lane "
                            f"{kin.lane_id or 'roadway'} for {self.intrusion_counts[tid]} frames"
                        ),
                    )
                )

        # Cleanup
        for tid in list(self.intrusion_counts.keys()):
            if tid not in active_tids:
                del self.intrusion_counts[tid]

        return candidates
