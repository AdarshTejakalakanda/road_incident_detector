"""Vehicle Response Detector Module.

Analyzes approaching vehicles entering a spatial Hazard Zone around an obstacle/incident,
tracking their kinematic behavior (speed change, deceleration, lateral swerve, heading change,
and lane change) and classifying their response.

Response classifications:
- NORMAL: Follows dominant path with steady speed and heading.
- SLOWING: Moderate deceleration/speed drop.
- BRAKING: Heavy deceleration or sudden emergency braking.
- LANE_CHANGE: Explicit lane transition or large lateral avoidance offset.
- EVASIVE_DRIFT: Sharp steering/swerve maneuver around obstacle without formal lane change.
- STOPPED: Approaching vehicle forced to come to a complete halt behind/adjacent to obstacle.
"""

import math
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, calculate_directional_cosine
from ..traffic_flow import SpatialTrafficFlowField

logger = logging.getLogger("road_surveillance.events.hazard.vehicle_response")


class VehicleResponseClass(str, Enum):
    NORMAL = "NORMAL"
    SLOWING = "SLOWING"
    BRAKING = "BRAKING"
    LANE_CHANGE = "LANE_CHANGE"
    EVASIVE_DRIFT = "EVASIVE_DRIFT"
    STOPPED = "STOPPED"


@dataclass
class VehicleReaction:
    """Record of an approaching vehicle's observed reaction to a hazard."""
    hazard_id: str
    vehicle_tid: int
    response: VehicleResponseClass
    speed_drop_ratio: float
    max_deceleration_ms2: float
    lateral_deviation_m: float
    heading_deviation_deg: float
    is_evasive: bool
    entry_speed_kmh: float
    min_speed_kmh: float
    entry_lane_id: Optional[str]
    current_lane_id: Optional[str]
    frame_idx: int
    timestamp_ns: int
    details: str = ""


@dataclass
class HazardZone:
    """Defines a spatial area of influence around an obstacle or stalled vehicle."""
    hazard_id: str
    ground_pos: Tuple[float, float]               # (x, y) in ground coords or normalized coords
    norm_pos: Tuple[float, float]                 # Normalized image coords (norm_x, norm_y)
    approach_radius_m: float = 30.0               # Upstream corridor radius to detect approaching traffic
    proximity_radius_m: float = 8.0               # Immediate zone around hazard
    is_calibrated: bool = False
    source_track_id: Optional[int] = None
    hazard_type: str = "generic"                  # "stalled_vehicle", "debris", "animal"


@dataclass
class VehicleApproachState:
    """Internal tracking state for a vehicle approaching a hazard zone."""
    vehicle_tid: int
    hazard_id: str
    entry_frame: int
    entry_pos: Tuple[float, float]
    entry_speed_kmh: float
    entry_lane_id: Optional[str]
    entry_heading_vec: Tuple[float, float]

    min_speed_kmh: float
    max_decel_ms2: float
    max_lateral_dev_m: float
    max_heading_dev_deg: float
    last_frame_updated: int
    stationary_frames: int = 0
    current_lane_id: Optional[str] = None
    last_pos: Tuple[float, float] = (0.0, 0.0)


class VehicleResponseDetector:
    """Tracks and classifies responses of vehicles approaching hazard zones."""

    def __init__(
        self,
        approach_radius_m: float = 30.0,
        approach_radius_norm: float = 0.35,
        min_approach_speed_kmh: float = 12.0,
        slowing_drop_threshold: float = 0.15,      # 15% speed drop
        braking_drop_threshold: float = 0.35,      # 35% speed drop
        braking_decel_threshold_ms2: float = 3.0,  # 3.0 m/s^2 deceleration
        braking_decel_pixel_threshold: float = 80.0,
        evasive_heading_deg: float = 18.0,         # 18 degrees heading swerve
        evasive_lateral_m: float = 1.4,            # 1.4 meters lateral displacement
        evasive_lateral_norm: float = 0.05,
    ):
        self.approach_radius_m = approach_radius_m
        self.approach_radius_norm = approach_radius_norm
        self.min_approach_speed_kmh = min_approach_speed_kmh
        self.slowing_drop_threshold = slowing_drop_threshold
        self.braking_drop_threshold = braking_drop_threshold
        self.braking_decel_threshold_ms2 = braking_decel_threshold_ms2
        self.braking_decel_pixel_threshold = braking_decel_pixel_threshold
        self.evasive_heading_deg = evasive_heading_deg
        self.evasive_lateral_m = evasive_lateral_m
        self.evasive_lateral_norm = evasive_lateral_norm

        # (hazard_id, tid) -> VehicleApproachState
        self._approach_states: Dict[Tuple[str, int], VehicleApproachState] = {}
        # hazard_id -> List[VehicleReaction]
        self._reactions_by_hazard: Dict[str, List[VehicleReaction]] = {}

    def reset(self) -> None:
        """Reset internal tracking states."""
        self._approach_states.clear()
        self._reactions_by_hazard.clear()

    def update(
        self,
        hazard_zones: List[HazardZone],
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        frame_idx: int,
        pts_ns: int,
        flow_field: Optional[SpatialTrafficFlowField] = None,
        frame_shape: Tuple[int, int] = (1080, 1920),
    ) -> Dict[str, List[VehicleReaction]]:
        """Evaluate all active tracks relative to active hazard zones."""
        H, W = frame_shape[:2]
        current_reactions: Dict[str, List[VehicleReaction]] = {hz.hazard_id: [] for hz in hazard_zones}
        active_hazard_ids = {hz.hazard_id for hz in hazard_zones}

        # Clean up stale approach states for retired hazards
        for key in list(self._approach_states.keys()):
            if key[0] not in active_hazard_ids:
                del self._approach_states[key]

        for hz in hazard_zones:
            h_id = hz.hazard_id
            if h_id not in self._reactions_by_hazard:
                self._reactions_by_hazard[h_id] = []

            for tid, tr in tracks.items():
                if tr.status not in ("active", "coasting"):
                    continue
                # Do not evaluate the hazard object itself as its own approaching vehicle
                if hz.source_track_id is not None and tid == hz.source_track_id:
                    continue
                if tr.get_effective_label() not in Track.VEHICLE_CLASSES:
                    continue

                kin = kinematics.get(tid)
                if not kin:
                    continue

                # Compute distance and relative geometry
                is_calib = hz.is_calibrated and kin.is_calibrated and kin.is_ground_valid
                gc_x, gc_y = Track.calculate_ground_contact(tr.current_bbox_xywh)
                norm_x = float(np.clip(gc_x / max(1.0, W), 0.0, 1.0))
                norm_y = float(np.clip(gc_y / max(1.0, H), 0.0, 1.0))

                if is_calib:
                    dx = hz.ground_pos[0] - kin.ground_pos[0]
                    dy = hz.ground_pos[1] - kin.ground_pos[1]
                    dist = math.hypot(dx, dy)
                    max_approach_r = hz.approach_radius_m
                    veh_pos = kin.ground_pos
                else:
                    dx = hz.norm_pos[0] - norm_x
                    dy = hz.norm_pos[1] - norm_y
                    dist = math.hypot(dx, dy)
                    max_approach_r = self.approach_radius_norm
                    veh_pos = (norm_x, norm_y)

                # Check if vehicle is within spatial approach zone
                if dist > max_approach_r:
                    # Vehicle outside approach zone; if it had an active approach state that completed, clean it up
                    if (h_id, tid) in self._approach_states:
                        state = self._approach_states[(h_id, tid)]
                        # Finalize reaction if vehicle spent >= 3 frames in approach zone
                        if (frame_idx - state.entry_frame) >= 3:
                            reaction = self._classify_reaction(state, hz, frame_idx, pts_ns)
                            self._record_reaction(reaction)
                            current_reactions[h_id].append(reaction)
                        del self._approach_states[(h_id, tid)]
                    continue

                # Vehicle is within approach distance! Verify it is approaching or within proximity
                # Heading dot relative displacement: dot(v, d) > 0 means closing in
                heading_u = kin.heading_vector
                dot_approach = heading_u[0] * dx + heading_u[1] * dy

                # If vehicle is moving away and was never tracked in approach state, ignore
                if dot_approach < -0.1 and (h_id, tid) not in self._approach_states:
                    continue

                # Update or initialize vehicle approach state
                state_key = (h_id, tid)
                if state_key not in self._approach_states:
                    # Initialize entry state
                    self._approach_states[state_key] = VehicleApproachState(
                        vehicle_tid=tid,
                        hazard_id=h_id,
                        entry_frame=frame_idx,
                        entry_pos=veh_pos,
                        entry_speed_kmh=max(kin.speed_kmh, self.min_approach_speed_kmh),
                        entry_lane_id=kin.lane_id,
                        entry_heading_vec=heading_u,
                        min_speed_kmh=kin.speed_kmh,
                        max_decel_ms2=max(0.0, -kin.acceleration_mag if kin.acceleration_mag < 0 else 0.0),
                        max_lateral_dev_m=0.0,
                        max_heading_dev_deg=0.0,
                        last_frame_updated=frame_idx,
                        stationary_frames=1 if kin.speed_kmh < 3.0 else 0,
                        current_lane_id=kin.lane_id,
                        last_pos=veh_pos,
                    )
                else:
                    # Update running kinematic statistics
                    state = self._approach_states[state_key]
                    state.last_frame_updated = frame_idx
                    state.current_lane_id = kin.lane_id
                    state.min_speed_kmh = min(state.min_speed_kmh, kin.speed_kmh)

                    # Deceleration
                    decel = max(0.0, -kin.acceleration_mag if kin.acceleration_mag < 0 else 0.0)
                    if not is_calib and kin.pixel_acceleration < 0:
                        pixel_decel = abs(kin.pixel_acceleration)
                        state.max_decel_ms2 = max(state.max_decel_ms2, pixel_decel / 25.0)  # ~approx scale
                    else:
                        state.max_decel_ms2 = max(state.max_decel_ms2, decel)

                    if kin.speed_kmh < 3.0:
                        state.stationary_frames += 1
                    else:
                        state.stationary_frames = 0

                    # Heading deviation relative to entry heading and flow
                    cos_entry = calculate_directional_cosine(heading_u, state.entry_heading_vec)
                    ang_entry_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_entry))))
                    state.max_heading_dev_deg = max(state.max_heading_dev_deg, ang_entry_deg)

                    # Lateral deviation relative to entry line of motion
                    u_ex, u_ey = state.entry_heading_vec
                    norm_perp = (-u_ey, u_ex)  # Perpendicular normal
                    disp_x = veh_pos[0] - state.entry_pos[0]
                    disp_y = veh_pos[1] - state.entry_pos[1]
                    lateral_offset = abs(disp_x * norm_perp[0] + disp_y * norm_perp[1])
                    state.max_lateral_dev_m = max(state.max_lateral_dev_m, lateral_offset)
                    state.last_pos = veh_pos

                    # Check for immediate evasive reactions (braking, stop, swerve, lane change)
                    lat_thresh = self.evasive_lateral_m if is_calib else self.evasive_lateral_norm
                    is_swerve = (
                        state.max_heading_dev_deg >= self.evasive_heading_deg
                        and state.max_lateral_dev_m >= lat_thresh
                    )
                    is_lane_change = (
                        state.entry_lane_id is not None
                        and state.current_lane_id is not None
                        and state.entry_lane_id != state.current_lane_id
                    )
                    is_severe_decel = state.max_decel_ms2 >= self.braking_decel_threshold_ms2
                    is_stopped = state.stationary_frames >= 5

                    if is_stopped or is_severe_decel or is_swerve or is_lane_change:
                        reaction = self._classify_reaction(state, hz, frame_idx, pts_ns)
                        if reaction.is_evasive:
                            self._record_reaction(reaction)
                            current_reactions[h_id].append(reaction)

        return current_reactions

    def _classify_reaction(
        self,
        state: VehicleApproachState,
        hz: HazardZone,
        frame_idx: int,
        pts_ns: int,
    ) -> VehicleReaction:
        """Classify vehicle response based on accumulated trajectory statistics."""
        entry_spd = max(1.0, state.entry_speed_kmh)
        spd_drop = max(0.0, (entry_spd - state.min_speed_kmh) / entry_spd)
        lane_changed = (
            state.entry_lane_id is not None
            and state.current_lane_id is not None
            and state.entry_lane_id != state.current_lane_id
        )

        is_calib = hz.is_calibrated
        lat_thresh = self.evasive_lateral_m if is_calib else self.evasive_lateral_norm

        response = VehicleResponseClass.NORMAL
        is_evasive = False
        details = "Normal traversal"

        # 1. STOPPED: Vehicle stopped behind/at hazard
        if state.stationary_frames >= 5 or state.min_speed_kmh < 2.5:
            response = VehicleResponseClass.STOPPED
            is_evasive = True
            details = f"Vehicle came to complete stop (v_min={state.min_speed_kmh:.1f} km/h)"

        # 2. EMERGENCY BRAKING: Heavy deceleration or massive speed drop
        elif state.max_decel_ms2 >= self.braking_decel_threshold_ms2 or spd_drop >= self.braking_drop_threshold:
            response = VehicleResponseClass.BRAKING
            is_evasive = True
            details = f"Emergency braking (decel={state.max_decel_ms2:.1f} m/s^2, drop={spd_drop*100:.0f}%)"

        # 3. LANE CHANGE: Avoidance lane switch
        elif lane_changed or state.max_lateral_dev_m >= (lat_thresh * 1.8):
            response = VehicleResponseClass.LANE_CHANGE
            is_evasive = True
            details = f"Avoidance lane change (lateral_dev={state.max_lateral_dev_m:.2f}, lanes: {state.entry_lane_id}->{state.current_lane_id})"

        # 4. EVASIVE DRIFT: Swerve / heading deflection around obstacle
        elif state.max_heading_dev_deg >= self.evasive_heading_deg and state.max_lateral_dev_m >= lat_thresh:
            response = VehicleResponseClass.EVASIVE_DRIFT
            is_evasive = True
            details = f"Evasive swerve (heading_dev={state.max_heading_dev_deg:.1f} deg, lateral_dev={state.max_lateral_dev_m:.2f})"

        # 5. SLOWING: Noticeable deceleration
        elif spd_drop >= self.slowing_drop_threshold or state.max_decel_ms2 >= 1.5:
            response = VehicleResponseClass.SLOWING
            is_evasive = True
            details = f"Noticeable slowing (drop={spd_drop*100:.0f}%, decel={state.max_decel_ms2:.1f} m/s^2)"

        return VehicleReaction(
            hazard_id=state.hazard_id,
            vehicle_tid=state.vehicle_tid,
            response=response,
            speed_drop_ratio=spd_drop,
            max_deceleration_ms2=state.max_decel_ms2,
            lateral_deviation_m=state.max_lateral_dev_m,
            heading_deviation_deg=state.max_heading_dev_deg,
            is_evasive=is_evasive,
            entry_speed_kmh=state.entry_speed_kmh,
            min_speed_kmh=state.min_speed_kmh,
            entry_lane_id=state.entry_lane_id,
            current_lane_id=state.current_lane_id,
            frame_idx=frame_idx,
            timestamp_ns=pts_ns,
            details=details,
        )

    def _record_reaction(self, reaction: VehicleReaction) -> None:
        h_id = reaction.hazard_id
        if h_id not in self._reactions_by_hazard:
            self._reactions_by_hazard[h_id] = []

        # Deduplicate: replace previous reaction from same vehicle if current one is more severe
        existing = [r for r in self._reactions_by_hazard[h_id] if r.vehicle_tid == reaction.vehicle_tid]
        if not existing:
            self._reactions_by_hazard[h_id].append(reaction)
        else:
            # Keep the highest severity reaction
            severity_order = {
                VehicleResponseClass.NORMAL: 0,
                VehicleResponseClass.SLOWING: 1,
                VehicleResponseClass.LANE_CHANGE: 2,
                VehicleResponseClass.EVASIVE_DRIFT: 3,
                VehicleResponseClass.BRAKING: 4,
                VehicleResponseClass.STOPPED: 5,
            }
            if severity_order.get(reaction.response, 0) >= severity_order.get(existing[0].response, 0):
                self._reactions_by_hazard[h_id] = [
                    r for r in self._reactions_by_hazard[h_id] if r.vehicle_tid != reaction.vehicle_tid
                ]
                self._reactions_by_hazard[h_id].append(reaction)

    def get_reactions_for_hazard(self, hazard_id: str) -> List[VehicleReaction]:
        """Return all recorded vehicle reactions for a given hazard."""
        return self._reactions_by_hazard.get(hazard_id, [])

    def get_evasive_reaction_count(self, hazard_id: str) -> int:
        """Return the count of distinct approaching vehicles exhibiting evasive reactions."""
        reactions = self.get_reactions_for_hazard(hazard_id)
        return sum(1 for r in reactions if r.is_evasive)

    def has_corroborated_evasive_maneuver(self, hazard_id: str, min_reactions: int = 1) -> bool:
        """Check if at least min_reactions approaching vehicles exhibited evasive maneuvers."""
        return self.get_evasive_reaction_count(hazard_id) >= min_reactions
