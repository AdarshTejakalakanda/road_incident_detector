"""OpenCV Geometry Engine Module."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from .filters import TrajectoryFilter
from .spatial_metrics import (
    TrackKinematics,
    TrajectoryConvergence,
    compute_trajectory_convergence,
)
from ..tracking.track_state import Track, TrackletState
from ..calibration.homography import PlanarHomography
from ..calibration.lane_manager import LaneManager


class GeometryEngine:
    """Calculates ground kinematics, spatial relations, and lane occupancy across tracks."""

    def __init__(
        self,
        homography: PlanarHomography,
        lane_manager: Optional[LaneManager] = None,
        smoothing_window: int = 5,
        stationary_speed_threshold_kmh: float = 3.0,
        is_camera_calibrated: Optional[bool] = None,
    ):
        self.homography = homography
        self.lane_manager = lane_manager or LaneManager()
        self.smoothing_window = smoothing_window
        self.stationary_speed_threshold_kmh = stationary_speed_threshold_kmh
        if is_camera_calibrated is not None:
            self.is_camera_calibrated = is_camera_calibrated
        else:
            self.is_camera_calibrated = getattr(homography, "is_calibrated", False)

        self.filter = TrajectoryFilter()
        self.stationary_timers: Dict[int, float] = {}  # track_id -> total stationary seconds
        self.prev_kinematics: Dict[int, TrackKinematics] = {}

    def reset(self) -> None:
        """Reset internal state between videos."""
        self.stationary_timers.clear()
        self.prev_kinematics.clear()

    def process_frame(
        self,
        tracklets: List[TrackletState],
        tracks: Dict[int, Track],
    ) -> Tuple[Dict[int, TrackKinematics], List[TrajectoryConvergence]]:
        """Compute ground kinematics and pairwise trajectory convergence for the current frame."""
        kinematics_map: Dict[int, TrackKinematics] = {}

        # 1. Compute individual track kinematics
        for tracklet in tracklets:
            tid = tracklet.track_id
            track = tracks.get(tid)
            if not track:
                continue

            # Ground contact pixel -> metric ground plane
            u, v = tracklet.ground_contact_px
            raw_X, raw_Y, is_valid_ground = self.homography.pixel_to_ground_validated(u, v)

            if not is_valid_ground and track.ground_history:
                # Horizon singularity: maintain last valid ground position rather than snapping to (0, 0)
                smooth_X, smooth_Y = track.ground_history[-1]
            else:
                # Trajectory filter
                smooth_X, smooth_Y = self.filter.filter_point((raw_X, raw_Y), track.ground_history)

            # Default kinematics
            vx, vy = 0.0, 0.0
            ax, ay = 0.0, 0.0
            speed_mps = 0.0
            speed_kmh = 0.0
            heading_deg = 0.0
            heading_vec = (0.0, 1.0)
            pixel_spd = 0.0
            pixel_acc = 0.0
            dt_sec = 0.0

            if len(track.ground_history) >= 1 and len(track.timestamps) >= 2:
                # Calculate dt from true nanosecond timestamps
                dt_ns = track.timestamps[-1] - track.timestamps[-2]
                dt_sec = dt_ns * 1e-9

                if dt_sec > 1e-5:
                    prev_X, prev_Y = track.ground_history[-1]
                    dx = smooth_X - prev_X
                    dy = smooth_Y - prev_Y
                    step_dist = float(np.hypot(dx, dy))

                    # Physical feasibility jump check: clamp impossible single-frame displacements
                    # (At 220 km/h = 61.1 m/s, max step in dt is 61.1 * dt + 2.0m tolerance)
                    max_allowed_step_m = (220.0 / 3.6) * dt_sec + 2.0
                    if step_dist > max_allowed_step_m and step_dist > 1e-4:
                        scale = max_allowed_step_m / step_dist
                        smooth_X = prev_X + dx * scale
                        smooth_Y = prev_Y + dy * scale
                        dx = smooth_X - prev_X
                        dy = smooth_Y - prev_Y

                    vx = dx / dt_sec
                    vy = dy / dt_sec
                    speed_mps = float(np.hypot(vx, vy))
                    speed_kmh = speed_mps * 3.6

                    if speed_mps > 0.5:
                        heading_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                        heading_vec = (vx / speed_mps, vy / speed_mps)
                    elif tid in self.prev_kinematics:
                        # Retain previous heading if vehicle stopped
                        heading_deg = self.prev_kinematics[tid].heading_deg
                        heading_vec = self.prev_kinematics[tid].heading_vector

                    # Physical bounds check: clamp impossible perspective singularities
                    MAX_ROAD_SPEED_KMH = 220.0
                    if speed_kmh > MAX_ROAD_SPEED_KMH:
                        speed_kmh = MAX_ROAD_SPEED_KMH
                        speed_mps = speed_kmh / 3.6
                        vx = heading_vec[0] * speed_mps
                        vy = heading_vec[1] * speed_mps

                    # Acceleration dv / dt
                    if tid in self.prev_kinematics:
                        prev_k = self.prev_kinematics[tid]
                        ax = (vx - prev_k.velocity_vector[0]) / dt_sec
                        ay = (vy - prev_k.velocity_vector[1]) / dt_sec

                    # Pixel-space invariant plane tracking
                    if len(track.position_history) >= 2:
                        prev_px = track.position_history[-2]
                        curr_px = track.position_history[-1]
                        px_dx = curr_px[0] - prev_px[0]
                        px_dy = curr_px[1] - prev_px[1]
                        pixel_spd = float(np.hypot(px_dx, px_dy)) / dt_sec
                        if tid in self.prev_kinematics:
                            pixel_acc = (pixel_spd - self.prev_kinematics[tid].pixel_speed) / dt_sec

            track.ground_history.append((smooth_X, smooth_Y))

            # Stationary timer accumulation
            if speed_kmh < self.stationary_speed_threshold_kmh and dt_sec > 0:
                self.stationary_timers[tid] = self.stationary_timers.get(tid, 0.0) + dt_sec
            else:
                self.stationary_timers[tid] = 0.0

            # Lane checks
            lane = self.lane_manager.get_lane_for_ground_point(smooth_X, smooth_Y)
            lane_id = lane.lane_id if lane else None
            is_active_lane = self.lane_manager.is_in_active_lane(smooth_X, smooth_Y)

            accel_mag = float(np.hypot(ax, ay))
            MAX_ROAD_ACCEL_MS2 = 60.0
            if accel_mag > MAX_ROAD_ACCEL_MS2:
                accel_mag = MAX_ROAD_ACCEL_MS2

            if tid in self.prev_kinematics:
                # Signed acceleration: negative if slowing down
                if speed_kmh < self.prev_kinematics[tid].speed_kmh:
                    accel_mag = -accel_mag

            kinematics = TrackKinematics(
                track_id=tid,
                frame_idx=tracklet.frame_idx,
                pts_ns=tracklet.pts_ns,
                ground_pos=(smooth_X, smooth_Y),
                velocity_vector=(vx, vy),
                speed_mps=speed_mps,
                speed_kmh=speed_kmh,
                acceleration_vector=(ax, ay),
                acceleration_mag=accel_mag,
                heading_deg=heading_deg,
                heading_vector=heading_vec,
                stationary_duration_sec=self.stationary_timers.get(tid, 0.0),
                lane_id=lane_id,
                is_in_active_lane=is_active_lane,
                is_ground_valid=is_valid_ground,
                pixel_speed=pixel_spd,
                pixel_acceleration=pixel_acc,
                is_calibrated=self.is_camera_calibrated,
            )

            kinematics_map[tid] = kinematics
            self.prev_kinematics[tid] = kinematics

        # 2. Pairwise convergence and inter-object distance
        convergences: List[TrajectoryConvergence] = []
        track_ids = list(kinematics_map.keys())

        for i in range(len(track_ids)):
            t1 = kinematics_map[track_ids[i]]
            for j in range(i + 1, len(track_ids)):
                t2 = kinematics_map[track_ids[j]]
                conv = compute_trajectory_convergence(
                    pos1=t1.ground_pos,
                    vel1=t1.velocity_vector,
                    pos2=t2.ground_pos,
                    vel2=t2.velocity_vector,
                )
                conv.track_id_1 = t1.track_id
                conv.track_id_2 = t2.track_id
                convergences.append(conv)

        # Cleanup timers for inactive tracks
        active_tids = set(kinematics_map.keys())
        for tid in list(self.stationary_timers.keys()):
            if tid not in active_tids:
                del self.stationary_timers[tid]
                self.prev_kinematics.pop(tid, None)

        return kinematics_map, convergences
