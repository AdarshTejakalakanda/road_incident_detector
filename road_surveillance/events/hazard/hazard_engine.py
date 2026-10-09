"""Shared Roadway Hazard Engine Module.

Orchestrates multi-modal hazard detection (Stalled Vehicles, Road Debris, Animal Intrusions)
using shared SpatialTrafficFlowField, Hazard Zones, and Approaching Vehicle Response Maneuvers.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from ..base import EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics
from ..traffic_flow import SpatialTrafficFlowField
from .vehicle_response import VehicleResponseDetector, HazardZone, VehicleReaction
from .stalled_vehicle_sm import StalledVehicleManager, StalledVehicleState, StalledVehicleSnapshot
from .debris_hazard import DebrisHazardModule
from .animal_hazard import AnimalHazardModule
from ...calibration.road_surface import RoadSurfaceDetector
from ...calibration.lane_manager import LaneManager

logger = logging.getLogger("road_surveillance.events.hazard.hazard_engine")


class HazardEngine:
    """Unified engine coordinating contextual road hazard detection."""

    def __init__(
        self,
        flow_field: Optional[SpatialTrafficFlowField] = None,
        min_stalled_duration_sec: float = 3.0,
        unassisted_stalled_duration_sec: float = 6.0,
        congestion_stopped_threshold: int = 2,
        cooldown_frames: int = 60,
        available_vocabulary: Optional[Dict[int, str]] = None,
    ):
        self.flow_field = flow_field or SpatialTrafficFlowField()
        self.cooldown_frames = cooldown_frames

        # Submodules
        self.response_detector = VehicleResponseDetector()
        self.stalled_manager = StalledVehicleManager(
            min_stationary_sec=min_stalled_duration_sec,
            unassisted_hazard_duration_sec=unassisted_stalled_duration_sec,
            congestion_vehicle_threshold=congestion_stopped_threshold,
        )
        self.debris_module = DebrisHazardModule(
            min_stationary_sec=min_stalled_duration_sec,
            available_vocabulary=available_vocabulary,
        )
        self.animal_module = AnimalHazardModule(
            available_vocabulary=available_vocabulary,
        )

        # Cooldown tracker: hazard_id -> last_emitted_frame
        self._last_emitted_frame: Dict[str, int] = {}

    def reset(self) -> None:
        """Reset submodules between video sessions."""
        self.response_detector.reset()
        self.stalled_manager.reset()
        self.debris_module.reset()
        self.animal_module.reset()
        self._last_emitted_frame.clear()

    def full_reset(self) -> None:
        """Completely reset including traffic flow models."""
        self.reset()
        self.flow_field.full_reset()

    def process_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_shape: Tuple[int, int] = (1080, 1920),
        road_surface: Optional[RoadSurfaceDetector] = None,
        lane_manager: Optional[LaneManager] = None,
        dt_sec: float = 0.033,
    ) -> List[EventCandidate]:
        """Process a surveillance video frame and detect all confirmed roadway hazards."""
        candidates: List[EventCandidate] = []
        H, W = frame_shape[:2]

        # Step 1: Update Traffic Flow Model from Active Moving Vehicles
        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue
            if tr.get_effective_label() not in Track.VEHICLE_CLASSES:
                continue
            kin = kinematics.get(tid)
            if not kin:
                continue

            if tr.track_age >= 3 and kin.speed_kmh >= 12.0:
                gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
                norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
                norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))

                if road_surface is None or road_surface.is_point_on_road(gc_x, gc_y, frame_shape):
                    dx, dy = kin.heading_vector
                    if np.hypot(dx, dy) > 0.5:
                        self.flow_field.observe_motion(norm_x, norm_y, (dx, dy), frame_idx)

        # Step 2: Assemble Candidate Hazard Zones
        hazard_zones: List[HazardZone] = []

        # 2a. Potential Stalled Vehicles (vehicles stationary or crawling)
        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue
            if tr.get_effective_label() not in Track.VEHICLE_CLASSES:
                continue
            kin = kinematics.get(tid)
            if not kin or kin.speed_kmh >= 10.0:
                continue

            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
            norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))
            is_calib = kin.is_calibrated and kin.is_ground_valid

            hazard_zones.append(
                HazardZone(
                    hazard_id=f"stalled_{tid}",
                    ground_pos=kin.ground_pos if is_calib else (norm_x, norm_y),
                    norm_pos=(norm_x, norm_y),
                    approach_radius_m=35.0,
                    proximity_radius_m=8.0,
                    is_calibrated=is_calib,
                    source_track_id=tid,
                    hazard_type="stalled_vehicle",
                )
            )

        # 2b. Debris hazard zones
        hazard_zones.extend(
            self.debris_module.identify_hazard_zones(tracks, kinematics, frame_shape, road_surface)
        )

        # 2c. Animal hazard zones
        hazard_zones.extend(
            self.animal_module.identify_hazard_zones(tracks, kinematics, frame_shape, road_surface)
        )

        # Step 3: Evaluate Approaching Vehicles around All Hazard Zones
        self.response_detector.update(
            hazard_zones=hazard_zones,
            tracks=tracks,
            kinematics=kinematics,
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            flow_field=self.flow_field,
            frame_shape=frame_shape,
        )

        # Step 4: Evaluate Stalled Vehicles via Contextual State Machine
        stalled_snapshots = self.stalled_manager.update(
            tracks=tracks,
            kinematics=kinematics,
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            flow_field=self.flow_field,
            response_detector=self.response_detector,
            frame_shape=frame_shape,
            dt_sec=dt_sec,
        )

        for snap in stalled_snapshots:
            h_key = f"stalled_{snap.track_id}"
            if (frame_idx - self._last_emitted_frame.get(h_key, -999)) >= self.cooldown_frames:
                self._last_emitted_frame[h_key] = frame_idx
                tr = tracks[snap.track_id]
                kin = kinematics[snap.track_id]

                # Get approaching vehicle reaction summary
                reactions = self.response_detector.get_reactions_for_hazard(h_key)
                evasive = [r for r in reactions if r.is_evasive]
                rx_str = ", ".join(f"T{r.vehicle_tid}:{r.response.value}" for r in evasive[:3]) or "None yet"

                cand = EventCandidate(
                    event_type="stalled_vehicle",
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp_ns=pts_ns,
                    track_ids=[snap.track_id] + [r.vehicle_tid for r in evasive],
                    primary_track_id=snap.track_id,
                    confidence=snap.confidence,
                    signals={
                        "stationary_duration_sec": snap.stationary_duration_sec,
                        "slow_duration_sec": snap.slow_duration_sec,
                        "is_in_active_lane": snap.is_in_corridor,
                        "neighbor_stopped_count": snap.neighbor_stopped_count,
                        "approaching_evasive_count": snap.approaching_evasive_count,
                        "reaction_summary": rx_str,
                        "state": snap.state.value,
                    },
                    ground_coordinates=kin.ground_pos,
                    lane_id=kin.lane_id,
                    description=(
                        f"Stalled vehicle hazard: Track {snap.track_id} ({tr.get_effective_label()}) "
                        f"stationary for {snap.stationary_duration_sec:.1f}s in active corridor "
                        f"(corroborated by {snap.approaching_evasive_count} evasive maneuvers: [{rx_str}])"
                    ),
                )
                candidates.append(cand)
                logger.info(
                    f"[{video_id}] STALLED VEHICLE HAZARD @ f={frame_idx}: Track {snap.track_id} "
                    f"stationary={snap.stationary_duration_sec:.1f}s, evasive_rx={snap.approaching_evasive_count}, conf={snap.confidence:.2f}"
                )

        # Step 5: Evaluate Debris Hazards
        debris_candidates = self.debris_module.evaluate_candidates(
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=tracks,
            kinematics=kinematics,
            response_detector=self.response_detector,
            flow_field=self.flow_field,
            frame_shape=frame_shape,
            road_surface=road_surface,
        )
        for dc in debris_candidates:
            h_key = f"debris_{dc.primary_track_id}"
            if (frame_idx - self._last_emitted_frame.get(h_key, -999)) >= self.cooldown_frames:
                self._last_emitted_frame[h_key] = frame_idx
                candidates.append(dc)

        # Step 6: Evaluate Animal Hazards
        animal_candidates = self.animal_module.evaluate_candidates(
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=tracks,
            kinematics=kinematics,
            response_detector=self.response_detector,
            flow_field=self.flow_field,
            frame_shape=frame_shape,
            road_surface=road_surface,
        )
        for ac in animal_candidates:
            h_key = f"animal_{ac.primary_track_id}"
            if (frame_idx - self._last_emitted_frame.get(h_key, -999)) >= self.cooldown_frames:
                self._last_emitted_frame[h_key] = frame_idx
                candidates.append(ac)

        return candidates


from ..base import CandidateDetector
from ...geometry.spatial_metrics import TrajectoryConvergence


class HazardCandidateDetector(CandidateDetector):
    """Adapter exposing HazardEngine as a standard CandidateDetector."""

    def __init__(self, hazard_engine: Optional[HazardEngine] = None, **kwargs):
        self.hazard_engine = hazard_engine or HazardEngine(**kwargs)

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
        return self.hazard_engine.process_frame(
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=tracks,
            kinematics=kinematics,
            lane_manager=lane_manager,
        )
