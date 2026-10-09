"""Road Debris / Stationary Obstacle Hazard Module.

Detects stationary obstacles on active roadway corridors and corroborates them using
spatial hazard zones and approaching vehicle response maneuvers (swerves, braking, lane changes).
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np

from ..base import EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics
from ..traffic_flow import SpatialTrafficFlowField
from .vehicle_response import VehicleResponseDetector, HazardZone, VehicleReaction
from ...calibration.road_surface import RoadSurfaceDetector

logger = logging.getLogger("road_surveillance.events.hazard.debris_hazard")


class DebrisHazardModule:
    """Detects and corroborates physical obstacles and debris on the active carriageway."""

    KNOWN_DEBRIS_CLASSES = {
        "backpack", "suitcase", "sports ball", "box", "package", "bottle", "debris", "tire"
    }

    NON_DEBRIS_CLASSES = {
        "person", "car", "motorcycle", "bus", "truck", "bicycle", "traffic light", "stop sign"
    }

    def __init__(
        self,
        min_stationary_sec: float = 3.0,
        min_reactions_to_confirm: int = 1,
        available_vocabulary: Optional[Dict[int, str]] = None,
    ):
        self.min_stationary_sec = min_stationary_sec
        self.min_reactions_to_confirm = min_reactions_to_confirm

        self.debris_vocabulary: Set[str] = set()
        if available_vocabulary:
            vocab_lower = {v.lower() for v in available_vocabulary.values()}
            self.debris_vocabulary = vocab_lower.intersection(self.KNOWN_DEBRIS_CLASSES)

        # Track persistent candidates: tid -> first_detected_frame
        self._stationary_since: Dict[int, float] = {}

    def reset(self) -> None:
        """Reset internal history."""
        self._stationary_since.clear()

    def identify_hazard_zones(
        self,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_shape: Tuple[int, int],
        road_surface: Optional[RoadSurfaceDetector] = None,
    ) -> List[HazardZone]:
        """Identify candidate debris objects and construct Hazard Zones for response tracking."""
        H, W = frame_shape[:2]
        zones: List[HazardZone] = []

        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue

            c_label = tr.get_effective_label().lower()
            if c_label in self.NON_DEBRIS_CLASSES:
                continue

            kin = kinematics.get(tid)
            if not kin:
                continue

            # Must be stationary or near-zero speed
            if kin.speed_kmh > 4.0:
                continue

            # Check road surface
            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            if road_surface is not None and not road_surface.is_point_on_road(gc_x, gc_y, frame_shape):
                continue

            norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
            norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))
            is_calib = kin.is_calibrated and kin.is_ground_valid

            zones.append(
                HazardZone(
                    hazard_id=f"debris_{tid}",
                    ground_pos=kin.ground_pos if is_calib else (norm_x, norm_y),
                    norm_pos=(norm_x, norm_y),
                    approach_radius_m=25.0,
                    proximity_radius_m=6.0,
                    is_calibrated=is_calib,
                    source_track_id=tid,
                    hazard_type="debris",
                )
            )

        return zones

    def evaluate_candidates(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        response_detector: VehicleResponseDetector,
        flow_field: Optional[SpatialTrafficFlowField],
        frame_shape: Tuple[int, int],
        road_surface: Optional[RoadSurfaceDetector] = None,
    ) -> List[EventCandidate]:
        """Correlate debris tracks with approaching vehicle reactions and emit event candidates."""
        candidates: List[EventCandidate] = []
        H, W = frame_shape[:2]

        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue

            c_label = tr.get_effective_label().lower()
            if c_label in self.NON_DEBRIS_CLASSES:
                continue

            kin = kinematics.get(tid)
            if not kin or kin.speed_kmh > 4.0:
                continue

            # Check road surface
            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            if road_surface is not None and not road_surface.is_point_on_road(gc_x, gc_y, frame_shape):
                continue

            norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
            norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))

            # Check active traffic corridor
            is_corridor = kin.is_in_active_lane
            if not is_corridor and flow_field is not None:
                is_corridor = flow_field.is_active_traffic_corridor(norm_x, norm_y)

            hazard_id = f"debris_{tid}"
            reactions = response_detector.get_reactions_for_hazard(hazard_id)
            evasive_reactions = [r for r in reactions if r.is_evasive]
            evasive_count = len(evasive_reactions)

            # Trigger condition:
            # 1. Known debris class + stationary on road, OR
            # 2. Stationary object with at least 1 evasive reaction from approaching traffic, OR
            # 3. Stationary in active corridor for >= min_stationary_sec
            is_known_debris = (c_label in self.debris_vocabulary) or (c_label in self.KNOWN_DEBRIS_CLASSES)
            has_corroboration = evasive_count >= self.min_reactions_to_confirm
            is_persistent_in_corridor = is_corridor and kin.stationary_duration_sec >= self.min_stationary_sec

            if is_known_debris or has_corroboration or is_persistent_in_corridor:
                # Calculate confidence
                conf = 0.50
                if is_known_debris:
                    conf += 0.20
                if has_corroboration:
                    conf += min(0.25, 0.15 * evasive_count)
                if is_corridor:
                    conf += 0.10
                conf = float(np.clip(conf, 0.50, 0.98))

                reaction_summary = ", ".join(
                    f"T{r.vehicle_tid}:{r.response.value}" for r in evasive_reactions[:3]
                ) or "None yet"

                cand = EventCandidate(
                    event_type="debris_hazard",
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp_ns=pts_ns,
                    track_ids=[tid] + [r.vehicle_tid for r in evasive_reactions],
                    primary_track_id=tid,
                    confidence=conf,
                    signals={
                        "detected_label": tr.get_effective_label(),
                        "stationary_duration_sec": kin.stationary_duration_sec,
                        "evasive_reactions_count": evasive_count,
                        "reaction_summary": reaction_summary,
                        "is_in_active_lane": is_corridor,
                        "corroborated_by_traffic": has_corroboration,
                    },
                    ground_coordinates=kin.ground_pos,
                    lane_id=kin.lane_id,
                    description=(
                        f"Road debris hazard: Track {tid} ({tr.get_effective_label()}) on roadway "
                        f"(stationary={kin.stationary_duration_sec:.1f}s, corroborated by {evasive_count} evasive maneuvers: [{reaction_summary}])"
                    ),
                )
                candidates.append(cand)
                logger.info(
                    f"[{video_id}] DEBRIS HAZARD @ f={frame_idx}: Track {tid} ({tr.get_effective_label()}), "
                    f"evasive_reactions={evasive_count}, conf={conf:.2f}"
                )

        return candidates
