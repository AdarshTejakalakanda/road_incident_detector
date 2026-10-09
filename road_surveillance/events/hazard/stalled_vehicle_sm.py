"""Stalled Vehicle State Machine Module.

Implements a contextual state machine to differentiate isolated roadway hazards from
traffic signals and congestion queues:

States:
  MOVING -> SLOWING -> SUSPICIOUSLY_SLOW -> STATIONARY -> ROADWAY_HAZARD
                                 │
                                 └──> NORMAL_STOP (traffic signal / congestion queue)
                                         │
                                         └──> (traffic clears, vehicle remains) -> ROADWAY_HAZARD
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics
from ..traffic_flow import SpatialTrafficFlowField
from .vehicle_response import VehicleResponseDetector

logger = logging.getLogger("road_surveillance.events.hazard.stalled_vehicle_sm")


class StalledVehicleState(str, Enum):
    MOVING = "MOVING"
    SLOWING = "SLOWING"
    SUSPICIOUSLY_SLOW = "SUSPICIOUSLY_SLOW"
    STATIONARY = "STATIONARY"
    NORMAL_STOP = "NORMAL_STOP"
    ROADWAY_HAZARD = "ROADWAY_HAZARD"


@dataclass
class StalledVehicleSnapshot:
    """Diagnostic snapshot of a vehicle's stalled state."""
    track_id: int
    state: StalledVehicleState
    speed_kmh: float
    stationary_duration_sec: float
    slow_duration_sec: float
    is_in_corridor: bool
    neighbor_stopped_count: int
    approaching_evasive_count: int
    confidence: float
    description: str


class StalledVehicleStateMachine:
    """State machine tracking a single vehicle's stalled hazard progression."""

    def __init__(
        self,
        track_id: int,
        min_slow_duration_sec: float = 1.5,
        min_stationary_duration_sec: float = 3.0,
        unassisted_hazard_duration_sec: float = 6.0,
        congestion_stopped_threshold: int = 2,
    ):
        self.track_id = track_id
        self.min_slow_duration_sec = min_slow_duration_sec
        self.min_stationary_duration_sec = min_stationary_duration_sec
        self.unassisted_hazard_duration_sec = unassisted_hazard_duration_sec
        self.congestion_stopped_threshold = congestion_stopped_threshold

        self.state: StalledVehicleState = StalledVehicleState.MOVING
        self.stationary_duration_sec: float = 0.0
        self.isolated_stationary_duration_sec: float = 0.0
        self.slow_duration_sec: float = 0.0
        self.state_entry_frame: int = 0
        self.last_updated_frame: int = 0
        self.hazard_emitted: bool = False
        self.evasive_approaching_count: int = 0
        self.neighbor_stopped_count: int = 0
        self.confidence: float = 0.0

    def step(
        self,
        kin: TrackKinematics,
        is_in_active_corridor: bool,
        neighbor_stopped_count: int,
        approaching_evasive_count: int,
        frame_idx: int,
        dt_sec: float = 0.033,
    ) -> StalledVehicleState:
        """Update vehicle state based on kinematics, corridor context, and surrounding responses."""
        self.last_updated_frame = frame_idx
        self.neighbor_stopped_count = neighbor_stopped_count
        self.evasive_approaching_count = approaching_evasive_count
        speed = kin.speed_kmh
        is_congestion = (neighbor_stopped_count >= self.congestion_stopped_threshold)

        # Update timers
        if speed < 2.5:
            self.stationary_duration_sec += dt_sec
            self.slow_duration_sec += dt_sec
            if not is_congestion:
                self.isolated_stationary_duration_sec += dt_sec
            else:
                self.isolated_stationary_duration_sec = 0.0
        elif speed < 10.0:
            self.stationary_duration_sec = 0.0
            self.isolated_stationary_duration_sec = 0.0
            self.slow_duration_sec += dt_sec
        else:
            self.stationary_duration_sec = 0.0
            self.isolated_stationary_duration_sec = 0.0
            self.slow_duration_sec = 0.0

        prev_state = self.state

        # State Transition Logic
        # Exit to MOVING: vehicle resumed active travel
        if speed >= 15.0:
            self.state = StalledVehicleState.MOVING
            self.hazard_emitted = False
            return self.state

        if self.state == StalledVehicleState.MOVING:
            if speed < 2.5:
                self.state = StalledVehicleState.SLOWING
            elif speed < 12.0 or (kin.acceleration_mag < -1.5):
                self.state = StalledVehicleState.SLOWING

        elif self.state == StalledVehicleState.SLOWING:
            if speed < 2.5:
                if self.stationary_duration_sec >= self.min_stationary_duration_sec:
                    self.state = StalledVehicleState.NORMAL_STOP if is_congestion else StalledVehicleState.STATIONARY
                elif self.slow_duration_sec >= self.min_slow_duration_sec and is_in_active_corridor:
                    self.state = StalledVehicleState.SUSPICIOUSLY_SLOW
            elif speed < 8.0:
                if self.slow_duration_sec >= self.min_slow_duration_sec and is_in_active_corridor:
                    self.state = StalledVehicleState.SUSPICIOUSLY_SLOW

        elif self.state == StalledVehicleState.SUSPICIOUSLY_SLOW:
            if is_congestion:
                self.state = StalledVehicleState.NORMAL_STOP
            elif speed < 2.5 and self.stationary_duration_sec >= self.min_stationary_duration_sec:
                self.state = StalledVehicleState.STATIONARY
            elif self.slow_duration_sec >= self.unassisted_hazard_duration_sec and is_in_active_corridor:
                # Crawling dangerously slow for a prolonged period in active corridor
                if approaching_evasive_count >= 1:
                    self.state = StalledVehicleState.ROADWAY_HAZARD

        elif self.state == StalledVehicleState.STATIONARY:
            if is_congestion:
                # Surrounding vehicles are also stopped (e.g. red traffic signal)
                self.state = StalledVehicleState.NORMAL_STOP
            elif not is_in_active_corridor and not kin.is_in_active_lane:
                # Pulled off to roadside shoulder/parking bay safely
                pass
            else:
                # In active traffic lane/corridor while traffic is otherwise flowing!
                # Corroboration condition:
                # a) At least 1 approaching vehicle exhibits evasive maneuver (braking, swerve, lane change), OR
                # b) Stalled unassisted in active corridor for >= unassisted_hazard_duration_sec
                if approaching_evasive_count >= 1 or self.isolated_stationary_duration_sec >= self.unassisted_hazard_duration_sec:
                    self.state = StalledVehicleState.ROADWAY_HAZARD

        elif self.state == StalledVehicleState.NORMAL_STOP:
            # Check if traffic cleared (congestion ended / signal turned green), but vehicle remains stuck
            if not is_congestion and speed < 2.5:
                if is_in_active_corridor or kin.is_in_active_lane:
                    self.state = StalledVehicleState.STATIONARY
            elif speed >= 12.0:
                self.state = StalledVehicleState.MOVING

        elif self.state == StalledVehicleState.ROADWAY_HAZARD:
            if is_congestion:
                self.state = StalledVehicleState.NORMAL_STOP
            elif speed >= 10.0:
                self.state = StalledVehicleState.MOVING

        if self.state != prev_state:
            self.state_entry_frame = frame_idx

        # Compute hazard confidence
        if self.state == StalledVehicleState.ROADWAY_HAZARD:
            base_conf = 0.70
            if approaching_evasive_count >= 1:
                base_conf += min(0.25, 0.15 * approaching_evasive_count)
            if self.isolated_stationary_duration_sec >= self.unassisted_hazard_duration_sec:
                base_conf += 0.10
            self.confidence = float(np.clip(base_conf, 0.70, 0.99))
        else:
            self.confidence = 0.0


        return self.state

    def get_snapshot(self) -> StalledVehicleSnapshot:
        """Return diagnostic snapshot."""
        return StalledVehicleSnapshot(
            track_id=self.track_id,
            state=self.state,
            speed_kmh=0.0,
            stationary_duration_sec=self.stationary_duration_sec,
            slow_duration_sec=self.slow_duration_sec,
            is_in_corridor=True,
            neighbor_stopped_count=self.neighbor_stopped_count,
            approaching_evasive_count=self.evasive_approaching_count,
            confidence=self.confidence,
            description=f"Track {self.track_id} in state {self.state.value} (stationary={self.stationary_duration_sec:.1f}s)",
        )


class StalledVehicleManager:
    """Manages state machines for all active surveillance vehicles."""

    def __init__(
        self,
        congestion_radius_m: float = 12.0,
        congestion_radius_norm: float = 0.15,
        congestion_vehicle_threshold: int = 2,
        min_stationary_sec: float = 3.0,
        unassisted_hazard_duration_sec: float = 6.0,
    ):
        self.congestion_radius_m = congestion_radius_m
        self.congestion_radius_norm = congestion_radius_norm
        self.congestion_vehicle_threshold = congestion_vehicle_threshold
        self.min_stationary_sec = min_stationary_sec
        self.unassisted_hazard_duration_sec = unassisted_hazard_duration_sec

        # track_id -> StalledVehicleStateMachine
        self._state_machines: Dict[int, StalledVehicleStateMachine] = {}

    def reset(self) -> None:
        """Reset state machines."""
        self._state_machines.clear()

    def update(
        self,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_idx: int,
        pts_ns: int,
        flow_field: Optional[SpatialTrafficFlowField],
        response_detector: Optional[VehicleResponseDetector],
        frame_shape: Tuple[int, int],
        dt_sec: float = 0.033,
    ) -> List[StalledVehicleSnapshot]:
        """Update state machines and return snapshots for all roadway hazard vehicles."""
        H, W = frame_shape[:2]
        hazard_snapshots: List[StalledVehicleSnapshot] = []
        active_tids = set()

        # Step 1: Identify all stationary / crawling vehicles for congestion clustering
        stopped_vehicles: List[Tuple[int, Tuple[float, float], bool]] = []
        for tid, tr in tracks.items():
            if tr.status not in ("active", "coasting"):
                continue
            if tr.get_effective_label() not in Track.VEHICLE_CLASSES:
                continue
            kin = kinematics.get(tid)
            if not kin:
                continue

            active_tids.add(tid)
            if kin.speed_kmh < 2.5:
                is_calib = kin.is_calibrated and kin.is_ground_valid
                pos = kin.ground_pos if is_calib else (
                    Track.calculate_ground_contact(tr.current_bbox_xywh)[0] / max(1.0, W),
                    Track.calculate_ground_contact(tr.current_bbox_xywh)[1] / max(1.0, H),
                )
                stopped_vehicles.append((tid, pos, is_calib))

        # Step 2: Update state machines for active vehicles
        for tid in active_tids:
            tr = tracks[tid]
            kin = kinematics.get(tid)
            if not kin:
                continue

            gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
            norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
            norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))

            # Check if location is in active corridor or active lane
            is_corridor = kin.is_in_active_lane
            if not is_corridor and flow_field is not None:
                is_corridor = flow_field.is_active_traffic_corridor(norm_x, norm_y)

            # Compute stopped neighbors
            is_calib = kin.is_calibrated and kin.is_ground_valid
            my_pos = kin.ground_pos if is_calib else (norm_x, norm_y)
            neighbor_stopped = 0
            for o_tid, o_pos, o_calib in stopped_vehicles:
                if o_tid == tid:
                    continue
                d = np.hypot(my_pos[0] - o_pos[0], my_pos[1] - o_pos[1])
                thresh = self.congestion_radius_m if is_calib else self.congestion_radius_norm
                if d <= thresh:
                    neighbor_stopped += 1

            # Check approaching evasive reactions for this vehicle
            hazard_id = f"stalled_{tid}"
            evasive_count = 0
            if response_detector is not None:
                evasive_count = response_detector.get_evasive_reaction_count(hazard_id)

            if tid not in self._state_machines:
                self._state_machines[tid] = StalledVehicleStateMachine(
                    track_id=tid,
                    min_stationary_duration_sec=self.min_stationary_sec,
                    unassisted_hazard_duration_sec=self.unassisted_hazard_duration_sec,
                    congestion_stopped_threshold=self.congestion_vehicle_threshold,
                )

            sm = self._state_machines[tid]
            state = sm.step(
                kin=kin,
                is_in_active_corridor=is_corridor,
                neighbor_stopped_count=neighbor_stopped,
                approaching_evasive_count=evasive_count,
                frame_idx=frame_idx,
                dt_sec=dt_sec,
            )

            if state == StalledVehicleState.ROADWAY_HAZARD:
                snapshot = sm.get_snapshot()
                snapshot.speed_kmh = kin.speed_kmh
                snapshot.is_in_corridor = is_corridor
                hazard_snapshots.append(snapshot)

        # Cleanup lost tracks
        for tid in list(self._state_machines.keys()):
            if tid not in active_tids:
                del self._state_machines[tid]

        return hazard_snapshots
