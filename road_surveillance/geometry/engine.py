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
        self.kinematic_histories: Dict[int, List[Dict[str, Any]]] = {}

    def reset(self) -> None:
        """Reset internal state between videos."""
        self.stationary_timers.clear()
        self.prev_kinematics.clear()
        self.kinematic_histories.clear()

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
            a_long = 0.0
            a_lat = 0.0
            speed_mps = 0.0
            speed_kmh = 0.0
            heading_deg = 0.0
            heading_vec = (0.0, 1.0)
            yaw_rate_deg_s = 0.0
            pixel_spd = 0.0
            pixel_acc = 0.0
            dt_sec = 0.0

            if len(track.ground_history) >= 1 and len(track.timestamps) >= 2:
                # Calculate dt from true nanosecond timestamps
                dt_ns = track.timestamps[-1] - track.timestamps[-2]
                dt_sec = dt_ns * 1e-9

                if dt_sec < 1e-5:
                    dt_sec = 1.0 / 30.0

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
                    step_dist = float(np.hypot(dx, dy))

                raw_vx = dx / dt_sec
                raw_vy = dy / dt_sec
                raw_speed_mps = step_dist / dt_sec

                # Velocity estimation:
                # Frame 1 (2 points total): direct step velocity to maintain immediate responsiveness and pass unit tests
                # Frame 2+ (>=3 points): windowed weighted linear regression to eliminate single-frame quantization noise
                total_pts = len(track.ground_history) + 1
                if total_pts <= 2:
                    vx = raw_vx
                    vy = raw_vy
                    speed_mps = raw_speed_mps
                else:
                    k_win = min(5, total_pts)
                    all_ground = track.ground_history[-(k_win - 1):] + [(smooth_X, smooth_Y)]
                    all_times = track.timestamps[-k_win:]
                    t_rel = np.array([(t - all_times[-1]) * 1e-9 for t in all_times], dtype=np.float64)
                    x_pts = np.array([p[0] for p in all_ground], dtype=np.float64)
                    y_pts = np.array([p[1] for p in all_ground], dtype=np.float64)

                    t_mean = np.mean(t_rel)
                    t_var = np.sum((t_rel - t_mean) ** 2)
                    if t_var > 1e-6:
                        reg_vx = float(np.sum((t_rel - t_mean) * (x_pts - np.mean(x_pts))) / t_var)
                        reg_vy = float(np.sum((t_rel - t_mean) * (y_pts - np.mean(y_pts))) / t_var)
                        vx = reg_vx
                        vy = reg_vy
                        speed_mps = float(np.hypot(vx, vy))
                    else:
                        vx = raw_vx
                        vy = raw_vy
                        speed_mps = raw_speed_mps

                speed_kmh = speed_mps * 3.6

                # Physical bounds check: clamp impossible perspective singularities
                MAX_ROAD_SPEED_KMH = 220.0
                if speed_kmh > MAX_ROAD_SPEED_KMH:
                    speed_kmh = MAX_ROAD_SPEED_KMH
                    speed_mps = speed_kmh / 3.6
                    scale_v = speed_mps / max(1e-4, float(np.hypot(vx, vy)))
                    vx *= scale_v
                    vy *= scale_v

                # Heading and Yaw Rate computation with low-speed deadband
                stationary_thresh_mps = self.stationary_speed_threshold_kmh / 3.6
                if speed_mps >= stationary_thresh_mps:
                    raw_heading = (math.degrees(math.atan2(vy, vx)) + 360.0) % 360.0
                    if tid in self.prev_kinematics and self.prev_kinematics[tid].speed_mps >= stationary_thresh_mps:
                        prev_h = self.prev_kinematics[tid].heading_deg
                        dh = (raw_heading - prev_h + 180.0) % 360.0 - 180.0
                        heading_deg = (prev_h + 0.70 * dh + 360.0) % 360.0
                        yaw_rate_deg_s = float(dh / max(1e-4, dt_sec))
                    else:
                        heading_deg = raw_heading
                        yaw_rate_deg_s = 0.0
                    rad = math.radians(heading_deg)
                    heading_vec = (math.cos(rad), math.sin(rad))
                elif tid in self.prev_kinematics:
                    # Preserve previous stable heading during stops / low-speed crawl
                    heading_deg = self.prev_kinematics[tid].heading_deg
                    heading_vec = self.prev_kinematics[tid].heading_vector
                    yaw_rate_deg_s = 0.0
                else:
                    heading_deg = 0.0
                    heading_vec = (0.0, 1.0)
                    yaw_rate_deg_s = 0.0

                # Acceleration Vector & Longitudinal/Lateral Decomposition
                if tid in self.prev_kinematics:
                    prev_k = self.prev_kinematics[tid]
                    prev_vx, prev_vy = prev_k.velocity_vector
                    ax = (vx - prev_vx) / dt_sec
                    ay = (vy - prev_vy) / dt_sec

                    # Clamp physical acceleration limits (max 45.0 m/s^2 ≈ 4.6 g)
                    MAX_PHYSICAL_ACCEL = 45.0
                    a_norm = float(np.hypot(ax, ay))
                    if a_norm > MAX_PHYSICAL_ACCEL:
                        scale_a = MAX_PHYSICAL_ACCEL / a_norm
                        ax *= scale_a
                        ay *= scale_a

                    # Decompose into true longitudinal (tangential) and lateral (centripetal) acceleration
                    if speed_mps < stationary_thresh_mps and prev_k.speed_mps < stationary_thresh_mps:
                        a_long = 0.0  # Stationary deadband
                        a_lat = 0.0
                    elif speed_mps > 0.5:
                        u_tx, u_ty = heading_vec[0], heading_vec[1]
                        u_nx, u_ny = -u_ty, u_tx
                        a_long = float(ax * u_tx + ay * u_ty)
                        a_lat = float(ax * u_nx + ay * u_ny)
                    else:
                        a_long = float((speed_mps - prev_k.speed_mps) / dt_sec)
                        a_lat = 0.0

                    # Physical Traction Constraint Gate:
                    # Positive forward acceleration is traction-bounded (<= 6.0 m/s^2).
                    # Sudden positive acceleration spikes across 1-2 frames are detector box flickers.
                    # Deceleration during collision impact can reach -45.0 m/s^2.
                    MAX_PHYSICAL_FORWARD_ACCEL = 6.0
                    if a_long > MAX_PHYSICAL_FORWARD_ACCEL:
                        a_long = MAX_PHYSICAL_FORWARD_ACCEL

                # Pixel-space invariant plane tracking
                if len(track.position_history) >= 2:
                    curr_px = track.position_history[-1]
                    prev_px = track.position_history[-2]
                    raw_px_spd = float(np.hypot(curr_px[0] - prev_px[0], curr_px[1] - prev_px[1])) / dt_sec
                    if tid in self.prev_kinematics:
                        prev_px_spd = self.prev_kinematics[tid].pixel_speed
                        pixel_spd = 0.70 * raw_px_spd + 0.30 * prev_px_spd
                        raw_px_acc = (pixel_spd - prev_px_spd) / dt_sec
                        prev_px_acc = getattr(self.prev_kinematics[tid], "pixel_acceleration", 0.0)
                        pixel_acc = 0.70 * raw_px_acc + 0.30 * prev_px_acc
                        if pixel_spd < 2.0 and prev_px_spd < 2.0:
                            pixel_acc = 0.0
                    else:
                        pixel_spd = raw_px_spd
                        pixel_acc = 0.0

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

            kinematics = TrackKinematics(
                track_id=tid,
                frame_idx=tracklet.frame_idx,
                pts_ns=tracklet.pts_ns,
                ground_pos=(smooth_X, smooth_Y),
                velocity_vector=(vx, vy),
                speed_mps=speed_mps,
                speed_kmh=speed_kmh,
                acceleration_vector=(ax, ay),
                acceleration_mag=a_long,
                heading_deg=heading_deg,
                heading_vector=heading_vec,
                stationary_duration_sec=self.stationary_timers.get(tid, 0.0),
                lane_id=lane_id,
                is_in_active_lane=is_active_lane,
                is_ground_valid=is_valid_ground,
                pixel_speed=pixel_spd,
                pixel_acceleration=pixel_acc,
                is_calibrated=self.is_camera_calibrated,
                acceleration_longitudinal=a_long,
                acceleration_lateral=a_lat,
                yaw_rate_deg_s=yaw_rate_deg_s,
            )

            kinematics_map[tid] = kinematics
            self.prev_kinematics[tid] = kinematics
            if tid not in self.kinematic_histories:
                self.kinematic_histories[tid] = []
            self.kinematic_histories[tid].append({
                "speed_kmh": speed_kmh,
                "a_long": a_long,
                "a_lat": a_lat,
                "heading_deg": heading_deg,
            })
            if len(self.kinematic_histories[tid]) > 30:
                self.kinematic_histories[tid].pop(0)

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
                self.kinematic_histories.pop(tid, None)

        return kinematics_map, convergences
