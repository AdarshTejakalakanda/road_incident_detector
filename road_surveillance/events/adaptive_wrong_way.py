"""Adaptive Wrong-Way Incident Detection Module.

Learns dominant traffic-flow direction dynamically from observed vehicle trajectories,
persists the flow model across traffic gaps, and flags vehicles moving persistently
against the learned flow without hardcoding camera orientation or lane directions.

Architecture:
Track State
   ↓
Road/Lane Region
   ↓
Learn dominant traffic-flow direction
   ↓
Persist learned flow model
   ↓
Compare individual vehicle trajectory
   ↓
Opposite-direction score
   ↓
Persistence + confidence checks
   ↓
Wrong-Way Incident
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from .base import EventCandidate
from ..tracking.track_state import Track
from ..geometry.spatial_metrics import TrackKinematics, calculate_directional_cosine
from ..calibration.road_surface import RoadSurfaceDetector

logger = logging.getLogger("road_surveillance.events.adaptive_wrong_way")


from .traffic_flow import FlowDirectionModel, SpatialTrafficFlowField

__all__ = ["FlowDirectionModel", "SpatialTrafficFlowField", "AdaptiveWrongWayModule"]


class AdaptiveWrongWayModule:
    """Adaptive Wrong-Way Incident Module operating on shared Track State."""

    def __init__(
        self,
        min_speed_kmh: float = 8.0,            # Vehicle must be moving at speed (8.0 km/h accommodates initial deceleration)
        min_track_age: int = 5,                # Require mature track
        cos_opposition_threshold: float = -0.60, # cos(theta) <= -0.60 (angle >= 127 deg)
        min_consecutive_frames: int = 8,       # Must persist across at least 8 frames per spec
        min_vehicle_confidence: float = 0.25,  # Reasonable detection confidence
        grid_rows: int = 4,
        grid_cols: int = 4,
    ):
        self.min_speed_kmh = min_speed_kmh
        self.min_track_age = min_track_age
        self.cos_opposition_threshold = cos_opposition_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.min_vehicle_confidence = min_vehicle_confidence

        self.flow_field = SpatialTrafficFlowField(grid_rows=grid_rows, grid_cols=grid_cols)
        # Tracking violations: track_id -> consecutive violation frame count
        self.violation_counts: Dict[int, int] = {}
        self.violation_cosines: Dict[int, List[float]] = {}
        # Active master wrong-way hazard events: track_id -> EventCandidate
        self.active_hazards: Dict[int, EventCandidate] = {}

    def reset(self) -> None:
        """Reset violation counters between videos while keeping models configurable."""
        self.violation_counts.clear()
        self.violation_cosines.clear()
        self.active_hazards.clear()
        self.flow_field.full_reset()

    def full_reset(self) -> None:
        """Completely reset flow field and violations."""
        self.violation_counts.clear()
        self.violation_cosines.clear()
        self.active_hazards.clear()
        self.flow_field.full_reset()

    def attach_near_miss(
        self,
        wrong_way_tid: int,
        other_tid: int,
        frame_idx: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attach near-miss observation of an oncoming vehicle passing the wrong-way vehicle."""
        if wrong_way_tid in self.active_hazards:
            hazard = self.active_hazards[wrong_way_tid]
            info = {
                "other_track_id": other_tid,
                "frame": frame_idx,
                **(details or {})
            }
            hazard.signals.setdefault("near_misses", []).append(info)
            return True
        return False

    def process_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_shape: Tuple[int, int],
        road_surface_detector: Optional[RoadSurfaceDetector] = None,
    ) -> List[EventCandidate]:
        """Process tracks and kinematics to learn traffic flow and flag wrong-way vehicles."""
        candidates: List[EventCandidate] = []
        H, W = frame_shape[:2]
        active_tids: Set[int] = set()

        # Step 1: Learn Dominant Traffic Flow from Stable Moving Vehicles
        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue
            eff_label = tr.get_effective_label()
            is_candidate_recovered = (
                eff_label == "candidate_vehicle_recovered"
                or tr.class_label == "candidate_vehicle_recovered"
                or getattr(tr, "audit_reclassified", False)
                or getattr(tr, "fused_class", "") == "candidate_vehicle_recovered"
            )
            if eff_label not in Track.VEHICLE_CLASSES or is_candidate_recovered:
                continue

            k = kinematics.get(tid)
            if not k:
                continue

            # Must have sufficient speed and maturity to establish dominant traffic flow
            if tr.track_age >= 3 and k.speed_kmh >= self.min_speed_kmh:
                gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
                if not (0.0 <= gc_x <= W and 0.0 <= gc_y <= H):
                    continue
                norm_x = float(gc_x / max(1.0, W))
                norm_y = float(gc_y / max(1.0, H))

                # Check road surface: vehicle must be on the road
                if road_surface_detector is None or road_surface_detector.is_point_on_road(gc_x, gc_y, frame_shape):
                    # Check heading validity
                    dx, dy = k.heading_vector
                    if math.hypot(dx, dy) > 1e-4:
                        # Only register into flow learning if NOT currently a persistent wrong-way offender
                        if not getattr(tr, "is_wrong_way", False) and self.violation_counts.get(tid, 0) < 3:
                            self.flow_field.observe_motion(norm_x, norm_y, (dx, dy), frame_idx, track_id=tid)

        # Step 2: Compare Individual Vehicle Trajectories Against Persisted Flow Model
        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue
            eff_label = tr.get_effective_label()
            is_candidate_recovered = (
                eff_label == "candidate_vehicle_recovered"
                or tr.class_label == "candidate_vehicle_recovered"
                or getattr(tr, "audit_reclassified", False)
                or getattr(tr, "fused_class", "") == "candidate_vehicle_recovered"
            )
            if eff_label not in Track.VEHICLE_CLASSES or is_candidate_recovered:
                continue

            k = kinematics.get(tid)
            if not k:
                continue

            active_tids.add(tid)

            # Gate 1: Track Maturity
            if tr.track_age < self.min_track_age:
                self.violation_counts[tid] = 0
                continue

            # Gate 2: Speed threshold (vehicle must be in active forward motion, not creeping or parking)
            if k.speed_kmh < self.min_speed_kmh:
                self.violation_counts[tid] = 0
                continue

            # Gate 3: Confidence threshold
            if tr.detection_confidence < self.min_vehicle_confidence:
                continue

            # Gate 4: Road Region Association
            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            if not (0.0 <= gc_x <= W and 0.0 <= gc_y <= H):
                self.violation_counts[tid] = 0
                continue
            if road_surface_detector is not None:
                if not road_surface_detector.is_point_on_road(gc_x, gc_y, frame_shape):
                    self.violation_counts[tid] = 0
                    continue

            norm_x = float(gc_x / max(1.0, W))
            norm_y = float(gc_y / max(1.0, H))

            # Retrieve persisted flow model (handles zero current traffic seamlessly)
            flow_model = self.flow_field.get_flow_model_for_position(norm_x, norm_y)
            if flow_model is None or not flow_model.is_persisted:
                # No mature flow direction established yet
                self.violation_counts[tid] = 0
                continue

            # Gate 5: Directional Opposition via normalized vector dot product
            # cos(alpha) = dot(v_track, v_flow) / (norm(v_track) * norm(v_flow) + epsilon)
            veh_dir = k.heading_vector
            cos_theta = calculate_directional_cosine(veh_dir, flow_model.direction_vector)

            if cos_theta <= self.cos_opposition_threshold:
                # Vehicle is traveling contrary to dominant flow
                self.violation_counts[tid] = self.violation_counts.get(tid, 0) + 1
                if tid not in self.violation_cosines:
                    self.violation_cosines[tid] = []
                self.violation_cosines[tid].append(cos_theta)
            else:
                self.violation_counts[tid] = 0
                if tid in self.violation_cosines:
                    self.violation_cosines[tid].clear()

            # Gate 6: Temporal Persistence
            curr_violations = self.violation_counts[tid]
            if curr_violations >= self.min_consecutive_frames:
                mean_cos = float(np.mean(self.violation_cosines[tid][-10:]))
                angle_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

                # Confirmed wrong-way status & audit telemetry on persistent Track
                tr.is_wrong_way = True
                tr.wrong_way_telemetry = {
                    "wrong_way_score": float(abs(mean_cos)),
                    "flow_vector": list(flow_model.direction_vector),
                    "track_vector": [float(veh_dir[0]), float(veh_dir[1])],
                    "angle_deviation": angle_deg,
                    "persistence_frames": curr_violations,
                    "lane_roadway_association": True,
                }

                cand_conf = float(np.clip(0.70 + 0.30 * abs(mean_cos) * flow_model.coherence, 0.70, 0.99))

                if tid not in self.active_hazards:
                    # Emit master wrong_way_hazard incident candidate
                    cand = EventCandidate(
                        event_type="wrong_way_hazard",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=cand_conf,
                        signals={
                            "directional_cosine": cos_theta,
                            "mean_opposition_cosine": mean_cos,
                            "cos_threshold": self.cos_opposition_threshold,
                            "consecutive_violation_frames": curr_violations,
                            "speed_kmh": k.speed_kmh,
                            "flow_direction": list(flow_model.direction_vector),
                            "flow_coherence": flow_model.coherence,
                            "flow_sample_count": flow_model.sample_count,
                            "vehicle_heading_vector": list(veh_dir),
                            "persisted_flow_model_active": True,
                            "near_misses": [],
                            "start_frame": frame_idx,
                            "end_frame": frame_idx,
                            "wrong_way_score": float(abs(mean_cos)),
                            "angle_deviation": angle_deg,
                        },
                        ground_coordinates=k.ground_pos,
                        description=(
                            f"Track {tid} ({tr.get_effective_label()}) traveling wrong-way at {k.speed_kmh:.1f} km/h "
                            f"(cos={cos_theta:.2f}, sustained for {curr_violations} frames against flow coherence {flow_model.coherence:.2f})"
                        ),
                    )
                    self.active_hazards[tid] = cand
                    candidates.append(cand)
                    logger.info(
                        f"[{video_id}] WRONG-WAY HAZARD @ f={frame_idx}: Track {tid} ({tr.get_effective_label()}) "
                        f"opposing traffic flow (cos={cos_theta:.2f}, speed={k.speed_kmh:.1f} km/h, persistence={curr_violations}f)"
                    )
                else:
                    # Master hazard already emitted: update temporal duration extent
                    hazard = self.active_hazards[tid]
                    hazard.signals["end_frame"] = frame_idx
                    hazard.signals["duration_frames"] = frame_idx - hazard.signals["start_frame"] + 1
                    hazard.signals["speed_kmh"] = k.speed_kmh
                    hazard.signals["consecutive_violation_frames"] = curr_violations

        # Cleanup disappeared tracks
        for tid in list(self.violation_counts.keys()):
            if tid not in active_tids:
                del self.violation_counts[tid]
                if tid in self.violation_cosines:
                    del self.violation_cosines[tid]
                if tid in self.active_hazards:
                    del self.active_hazards[tid]

        return candidates
from .base import CandidateDetector
from ..geometry.spatial_metrics import TrajectoryConvergence
from ..calibration.lane_manager import LaneManager


class AdaptiveWrongWayCandidateDetector(CandidateDetector):
    """Adapter exposing AdaptiveWrongWayModule as a standard CandidateDetector."""

    def __init__(self, adaptive_module: Optional[AdaptiveWrongWayModule] = None, **kwargs):
        self.module = adaptive_module or AdaptiveWrongWayModule(**kwargs)
        self.frame_shape: Tuple[int, int] = (720, 1280)

    def set_resolution(self, width: float, height: float) -> None:
        self.frame_shape = (int(height), int(width))

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
        return self.module.process_frame(
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=tracks,
            kinematics=kinematics,
            frame_shape=self.frame_shape,
        )

    def attach_near_miss(
        self,
        wrong_way_tid: int,
        other_tid: int,
        frame_idx: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self.module.attach_near_miss(wrong_way_tid, other_tid, frame_idx, details)

    def reset(self) -> None:
        self.module.reset()
