"""Animal Intrusion Roadway Hazard Module.

Detects animals intruding upon active roadway corridors and tracks approaching vehicle
evasive maneuvers to verify roadway risk.
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

logger = logging.getLogger("road_surveillance.events.hazard.animal_hazard")


class AnimalHazardModule:
    """Detects animal intrusions on active carriageways and corroborates safety risk."""

    DEFAULT_ANIMAL_CLASSES = {
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "animal"
    }

    def __init__(
        self,
        min_consecutive_frames: int = 4,
        available_vocabulary: Optional[Dict[int, str]] = None,
    ):
        self.min_consecutive_frames = min_consecutive_frames
        self.supported_animal_labels: Set[str] = set()

        if available_vocabulary:
            vocab_lower = {v.lower() for v in available_vocabulary.values()}
            self.supported_animal_labels = vocab_lower.intersection(self.DEFAULT_ANIMAL_CLASSES)

        self._consecutive_on_road: Dict[int, int] = {}

    def reset(self) -> None:
        """Reset internal history."""
        self._consecutive_on_road.clear()

    def identify_hazard_zones(
        self,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_shape: Tuple[int, int],
        road_surface: Optional[RoadSurfaceDetector] = None,
    ) -> List[HazardZone]:
        """Construct hazard zones around animals on the roadway."""
        H, W = frame_shape[:2]
        zones: List[HazardZone] = []

        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue

            c_label = tr.get_effective_label().lower()
            is_animal = (c_label in self.DEFAULT_ANIMAL_CLASSES) or (c_label in self.supported_animal_labels)
            if not is_animal:
                continue

            kin = kinematics.get(tid)
            if not kin:
                continue

            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            if road_surface is not None and not road_surface.is_point_on_road(gc_x, gc_y, frame_shape):
                continue

            norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
            norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))
            is_calib = kin.is_calibrated and kin.is_ground_valid

            zones.append(
                HazardZone(
                    hazard_id=f"animal_{tid}",
                    ground_pos=kin.ground_pos if is_calib else (norm_x, norm_y),
                    norm_pos=(norm_x, norm_y),
                    approach_radius_m=35.0,
                    proximity_radius_m=10.0,
                    is_calibrated=is_calib,
                    source_track_id=tid,
                    hazard_type="animal",
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
        """Correlate animal presence with approaching vehicle reactions and emit event candidates."""
        candidates: List[EventCandidate] = []
        H, W = frame_shape[:2]
        active_tids = set()

        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue

            c_label = tr.get_effective_label().lower()
            is_animal = (c_label in self.DEFAULT_ANIMAL_CLASSES) or (c_label in self.supported_animal_labels)
            if not is_animal:
                continue

            kin = kinematics.get(tid)
            if not kin:
                continue

            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            on_road = True
            if road_surface is not None:
                on_road = road_surface.is_point_on_road(gc_x, gc_y, frame_shape)

            if not on_road:
                self._consecutive_on_road[tid] = 0
                continue

            active_tids.add(tid)
            self._consecutive_on_road[tid] = self._consecutive_on_road.get(tid, 0) + 1
            frames_on_road = self._consecutive_on_road[tid]

            hazard_id = f"animal_{tid}"
            reactions = response_detector.get_reactions_for_hazard(hazard_id)
            evasive_reactions = [r for r in reactions if r.is_evasive]
            evasive_count = len(evasive_reactions)

            if frames_on_road >= self.min_consecutive_frames or evasive_count >= 1:
                conf = 0.60
                if frames_on_road >= 10:
                    conf += 0.15
                if evasive_count >= 1:
                    conf += min(0.20, 0.10 * evasive_count)
                conf = float(np.clip(conf, 0.60, 0.95))

                reaction_summary = ", ".join(
                    f"T{r.vehicle_tid}:{r.response.value}" for r in evasive_reactions[:3]
                ) or "None yet"

                cand = EventCandidate(
                    event_type="animal_hazard",
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp_ns=pts_ns,
                    track_ids=[tid] + [r.vehicle_tid for r in evasive_reactions],
                    primary_track_id=tid,
                    confidence=conf,
                    signals={
                        "animal_class": tr.get_effective_label(),
                        "consecutive_frames_on_road": frames_on_road,
                        "evasive_reactions_count": evasive_count,
                        "reaction_summary": reaction_summary,
                    },
                    ground_coordinates=kin.ground_pos,
                    lane_id=kin.lane_id,
                    description=(
                        f"Animal hazard: Track {tid} ({tr.get_effective_label()}) on active roadway "
                        f"for {frames_on_road} frames, {evasive_count} evasive traffic reactions: [{reaction_summary}]"
                    ),
                )
                candidates.append(cand)
                logger.info(
                    f"[{video_id}] ANIMAL HAZARD @ f={frame_idx}: Track {tid} ({tr.get_effective_label()}), "
                    f"persistence={frames_on_road}f, evasive_reactions={evasive_count}, conf={conf:.2f}"
                )

        # Cleanup lost tracks
        for tid in list(self._consecutive_on_road.keys()):
            if tid not in active_tids:
                del self._consecutive_on_road[tid]

        return candidates
