"""Stalled Vehicle Incident Candidate Detector."""

from typing import Dict, List
import numpy as np

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager


class StalledVehicleDetector(CandidateDetector):
    """Detects stalled vehicles on active carriageways and differentiates traffic congestion."""

    VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "van", "train"}

    def __init__(
        self,
        min_stationary_sec: float = 3.5,
        congestion_radius_m: float = 10.0,
        congestion_vehicle_threshold: int = 3,
        min_track_age: int = 8,
    ):
        self.min_stationary_sec = min_stationary_sec
        self.congestion_radius_m = congestion_radius_m
        self.congestion_vehicle_threshold = congestion_vehicle_threshold
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

        # Find all stationary vehicles
        stationary_vehicles = []
        for tid, kin in kinematics.items():
            tr = tracks.get(tid)
            if not tr or tr.class_label.lower() not in self.VEHICLE_LABELS:
                continue
            if tr.track_age >= self.min_track_age and kin.stationary_duration_sec >= self.min_stationary_sec:
                if kin.is_in_active_lane:
                    stationary_vehicles.append((tid, tr, kin))

        for tid, tr, kin in stationary_vehicles:
            # Check for traffic congestion cluster
            neighbor_stationary_count = 0
            for other_tid, _, other_kin in stationary_vehicles:
                if other_tid == tid:
                    continue
                d = float(np.hypot(kin.ground_pos[0] - other_kin.ground_pos[0], kin.ground_pos[1] - other_kin.ground_pos[1]))
                if d <= self.congestion_radius_m:
                    neighbor_stationary_count += 1

            is_congestion = neighbor_stationary_count >= self.congestion_vehicle_threshold

            if not is_congestion:
                candidates.append(
                    EventCandidate(
                        event_type="stalled_vehicle",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=min(0.95, 0.6 + 0.05 * (kin.stationary_duration_sec - self.min_stationary_sec)),
                        signals={
                            "stationary_duration_sec": kin.stationary_duration_sec,
                            "min_stationary_threshold_sec": self.min_stationary_sec,
                            "speed_kmh": kin.speed_kmh,
                            "is_in_active_lane": kin.is_in_active_lane,
                            "neighbor_stationary_vehicles": neighbor_stationary_count,
                            "congestion_detected": False,
                        },
                        ground_coordinates=kin.ground_pos,
                        lane_id=kin.lane_id,
                        description=(
                            f"Stalled vehicle candidate: Track {tid} ({tr.class_label}) stationary for "
                            f"{kin.stationary_duration_sec:.1f}s in active lane {kin.lane_id or 'roadway'}"
                        ),
                    )
                )

        return candidates
