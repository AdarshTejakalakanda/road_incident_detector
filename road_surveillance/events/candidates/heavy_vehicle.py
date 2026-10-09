"""Stranded Heavy Vehicle Incident Candidate Detector."""

from typing import Dict, List

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager


class HeavyVehicleDetector(CandidateDetector):
    """Detects stalled or stranded heavy commercial vehicles (trucks, buses) in active lanes.
    Explicitly scoped: does not infer jackknifing or cargo spills in Phase 1.
    """

    HEAVY_LABELS = {"truck", "bus"}

    def __init__(self, min_stationary_sec: float = 3.5, min_track_age: int = 8):
        self.min_stationary_sec = min_stationary_sec
        self.min_track_age = min_track_age

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
            if not tr or tr.class_label.lower() not in self.HEAVY_LABELS:
                continue

            if tr.track_age >= self.min_track_age and kin.stationary_duration_sec >= self.min_stationary_sec:
                if kin.is_in_active_lane:
                    candidates.append(
                        EventCandidate(
                            event_type="stranded_heavy_vehicle",
                            video_id=video_id,
                            frame_idx=frame_idx,
                            timestamp_ns=pts_ns,
                            track_ids=[tid],
                            primary_track_id=tid,
                            confidence=min(0.95, 0.7 + 0.05 * (kin.stationary_duration_sec - self.min_stationary_sec)),
                            signals={
                                "vehicle_class": tr.class_label,
                                "stationary_duration_sec": kin.stationary_duration_sec,
                                "is_in_active_lane": kin.is_in_active_lane,
                                "phase_1_scope_note": "Jackknifing, cargo spills, or mechanical failures are not inferred in Phase 1.",
                            },
                            ground_coordinates=kin.ground_pos,
                            lane_id=kin.lane_id,
                            description=(
                                f"Stranded heavy vehicle candidate: Track {tid} ({tr.class_label}) stationary for "
                                f"{kin.stationary_duration_sec:.1f}s in active lane {kin.lane_id or 'roadway'}"
                            ),
                        )
                    )

        return candidates
