"""Event Engine Module coordinating all candidate detectors."""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
from .base import CandidateDetector, EventCandidate
from .candidates.wrong_way import WrongWayDetector
from .adaptive_wrong_way import AdaptiveWrongWayCandidateDetector
from .candidates.collision import CollisionCandidateDetector
from .candidates.single_vehicle import SingleVehicleAnomalyDetector
from .candidates.distant_conflict import DistantTrajectoryConflictDetector
from .candidates.stalled_vehicle import StalledVehicleDetector
from .candidates.pedestrian import PedestrianHazardDetector
from .candidates.animal import AnimalHazardDetector
from .candidates.debris import DebrisCandidateDetector
from .candidates.heavy_vehicle import HeavyVehicleDetector

from ..tracking.track_state import Track
from ..geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ..calibration.lane_manager import LaneManager
from ..config.config_loader import EventsConfig

logger = logging.getLogger("road_surveillance.events.engine")


class EventEngine:
    """Orchestrates candidate incident detection across all core categories."""

    def __init__(
        self,
        config: Optional[EventsConfig] = None,
        available_vocabulary: Optional[Dict[int, str]] = None,
        cooldown_seconds: float = 10.0,
        is_camera_calibrated: bool = False,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.is_camera_calibrated = is_camera_calibrated
        self.detectors: List[CandidateDetector] = []
        self.last_event_time: Dict[str, float] = {}  # key -> last trigger timestamp_sec
        self.listeners: List[Callable[[EventCandidate], None]] = []

        cfg = config or EventsConfig()

        self.wrong_way_detector = None
        if cfg.wrong_way.enabled:
            self.wrong_way_detector = AdaptiveWrongWayCandidateDetector(
                min_speed_kmh=getattr(cfg.wrong_way, "min_speed_kmh", 8.0),
                cos_opposition_threshold=cfg.wrong_way.cos_threshold if cfg.wrong_way.cos_threshold <= -0.60 else -0.60,
                min_consecutive_frames=max(8, getattr(cfg.wrong_way, "min_consecutive_frames", 8)),
            )
            self.detectors.append(self.wrong_way_detector)

        if cfg.collision.enabled:
            col_det = CollisionCandidateDetector(
                is_camera_calibrated=self.is_camera_calibrated,
                distance_decay_threshold_m=cfg.collision.distance_decay_threshold_m,
                tca_threshold_sec=cfg.collision.tca_threshold_sec,
                deceleration_threshold_ms2=cfg.collision.deceleration_threshold_ms2,
                post_proximity_distance_m=cfg.collision.post_proximity_distance_m,
                iou_overlap_threshold=cfg.collision.iou_overlap_threshold,
                min_persistence_frames=getattr(cfg.collision, 'min_persistence_frames', 2),
                min_track_age=getattr(cfg.collision, 'min_track_age', 8),
            )
            col_det.wrong_way_handler = self.wrong_way_detector
            self.detectors.append(col_det)

        if getattr(cfg, "single_vehicle", None) and cfg.single_vehicle.enabled:
            self.detectors.append(
                SingleVehicleAnomalyDetector(
                    is_camera_calibrated=self.is_camera_calibrated,
                    video_warmup_frames=cfg.single_vehicle.video_warmup_frames,
                    min_track_age=cfg.single_vehicle.min_track_age,
                    deceleration_threshold_ms2=cfg.single_vehicle.deceleration_threshold_ms2,
                    severe_impact_decel_ms2=cfg.single_vehicle.severe_impact_decel_ms2,
                    yaw_rate_threshold_deg=cfg.single_vehicle.yaw_rate_threshold_deg,
                    aspect_ratio_change_threshold=cfg.single_vehicle.aspect_ratio_change_threshold,
                    min_persistence_frames=cfg.single_vehicle.min_persistence_frames,
                )
            )

        if getattr(cfg, "distant_conflict", None) and cfg.distant_conflict.enabled:
            dist_det = DistantTrajectoryConflictDetector(
                is_camera_calibrated=self.is_camera_calibrated,
                video_warmup_frames=cfg.distant_conflict.video_warmup_frames,
                min_track_age=cfg.distant_conflict.min_track_age,
                deceleration_threshold_ms2=cfg.distant_conflict.deceleration_threshold_ms2,
                tca_max_sec=cfg.distant_conflict.tca_max_sec,
                dca_norm_max=cfg.distant_conflict.dca_norm_max,
                dist_norm_max=cfg.distant_conflict.dist_norm_max,
                min_closing_speed_kmh=cfg.distant_conflict.min_closing_speed_kmh,
                min_persistence_frames=cfg.distant_conflict.min_persistence_frames,
            )
            dist_det.wrong_way_handler = self.wrong_way_detector
            self.detectors.append(dist_det)

        if cfg.stalled_vehicle.enabled:
            self.detectors.append(
                StalledVehicleDetector(
                    min_stationary_sec=cfg.stalled_vehicle.min_stationary_sec,
                    congestion_radius_m=cfg.stalled_vehicle.congestion_radius_m,
                    congestion_vehicle_threshold=cfg.stalled_vehicle.congestion_vehicle_threshold,
                )
            )

        if cfg.pedestrian_hazard.enabled:
            self.detectors.append(
                PedestrianHazardDetector(
                    min_consecutive_frames=cfg.pedestrian_hazard.min_consecutive_frames,
                )
            )

        if cfg.animal_hazard.enabled:
            self.detectors.append(
                AnimalHazardDetector(
                    min_consecutive_frames=cfg.animal_hazard.min_consecutive_frames,
                    available_vocabulary=available_vocabulary,
                )
            )

        if cfg.debris_candidate.enabled:
            self.detectors.append(
                DebrisCandidateDetector(
                    min_stationary_sec=cfg.debris_candidate.min_stationary_sec,
                    available_vocabulary=available_vocabulary,
                )
            )

        if cfg.heavy_vehicle.enabled:
            self.detectors.append(
                HeavyVehicleDetector(
                    min_stationary_sec=cfg.heavy_vehicle.min_stationary_sec,
                )
            )

        logger.info(f"EventEngine initialized with {len(self.detectors)} candidate detectors.")

    def register_listener(self, callback: Callable[[EventCandidate], None]) -> None:
        """Register a callback listener for new event candidates."""
        self.listeners.append(callback)

    def reset(self) -> None:
        """Reset event timers and detectors between videos."""
        self.last_event_time.clear()
        for d in self.detectors:
            if hasattr(d, "reset"):
                d.reset()

    def set_resolution(self, width: float, height: float) -> None:
        """Propagate video frame resolution to candidate detectors."""
        for d in self.detectors:
            if hasattr(d, "set_resolution"):
                d.set_resolution(width, height)

    def process_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
        lane_manager: LaneManager,
    ) -> List[EventCandidate]:
        """Run all candidate detectors and return deduplicated candidate events."""
        current_time_sec = pts_ns * 1e-9
        triggered_candidates: List[EventCandidate] = []

        for detector in self.detectors:
            candidates = detector.evaluate(
                frame_idx=frame_idx,
                pts_ns=pts_ns,
                video_id=video_id,
                tracks=tracks,
                kinematics=kinematics,
                convergences=convergences,
                lane_manager=lane_manager,
            )

            for cand in candidates:
                # Deduplication key: event_type + sorted track_ids
                etype = cand.event_type.value if hasattr(cand.event_type, "value") else str(cand.event_type)
                key = f"{etype}_{sorted(cand.track_ids)}"
                last_time = self.last_event_time.get(key, -999.0)

                if (current_time_sec - last_time) >= self.cooldown_seconds:
                    self.last_event_time[key] = current_time_sec
                    triggered_candidates.append(cand)
                    logger.warning(f"EVENT CANDIDATE TRIGGERED: [{cand.event_type}] {cand.description}")

                    # Notify listeners
                    for listener in self.listeners:
                        try:
                            listener(cand)
                        except Exception as e:
                            logger.error(f"Error in event listener: {e}")

        return triggered_candidates

    def get_suppressed_events(self) -> List[Dict[str, Any]]:
        """Collect all suppressed events across active detectors."""
        suppressed: List[Dict[str, Any]] = []
        for d in self.detectors:
            if hasattr(d, "suppressed_events"):
                suppressed.extend(d.suppressed_events)
        return suppressed

