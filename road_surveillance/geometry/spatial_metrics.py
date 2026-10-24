"""Spatial and Kinematic Metrics Computation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class TrackKinematics:
    """Calculated kinematic and geometric state for an active tracklet."""
    track_id: int
    frame_idx: int
    pts_ns: int
    ground_pos: Tuple[float, float]
    velocity_vector: Tuple[float, float]      # (vx, vy) in m/s
    speed_mps: float
    speed_kmh: float                         # speed_mps * 3.6
    acceleration_vector: Tuple[float, float]  # (ax, ay) in m/s^2
    acceleration_mag: float                  # Signed longitudinal deceleration/acceleration
    heading_deg: float                       # 0 to 360 degrees
    heading_vector: Tuple[float, float]      # Unit direction vector (ux, uy)
    stationary_duration_sec: float
    lane_id: Optional[str]
    is_in_active_lane: bool
    is_ground_valid: bool = True
    pixel_speed: float = 0.0                 # Speed in pixels per second
    pixel_acceleration: float = 0.0          # Signed acceleration in px/sec^2
    is_calibrated: bool = False              # Whether based on surveyed metric calibration
    acceleration_longitudinal: float = 0.0   # Signed tangential acceleration (dv/dt in m/s^2, <0 = braking)
    acceleration_lateral: float = 0.0        # Normal centripetal acceleration (swerving in m/s^2)
    yaw_rate_deg_s: float = 0.0              # Heading rotation rate (dtheta/dt in deg/s)


@dataclass
class TrajectoryConvergence:
    """Pairwise trajectory convergence metrics between two tracks."""
    track_id_1: int
    track_id_2: int
    distance_m: float
    is_converging: bool
    tca_sec: float  # Time-to-Closest-Approach in seconds
    dca_m: float    # Distance-at-Closest-Approach in meters


def compute_trajectory_convergence(
    pos1: Tuple[float, float],
    vel1: Tuple[float, float],
    pos2: Tuple[float, float],
    vel2: Tuple[float, float],
) -> TrajectoryConvergence:
    """Calculate Time-to-Closest-Approach (TCA) and Distance-at-Closest-Approach (DCA)."""
    p1 = np.array(pos1, dtype=np.float64)
    v1 = np.array(vel1, dtype=np.float64)
    p2 = np.array(pos2, dtype=np.float64)
    v2 = np.array(vel2, dtype=np.float64)

    rel_pos = p2 - p1
    rel_vel = v2 - v1

    current_dist = float(np.linalg.norm(rel_pos))
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))

    if rel_speed_sq < 1e-4:
        # Stationary relative to each other
        return TrajectoryConvergence(
            track_id_1=0,
            track_id_2=0,
            distance_m=current_dist,
            is_converging=False,
            tca_sec=float("inf"),
            dca_m=current_dist,
        )

    # TCA = - (r . v) / (v . v)
    tca = -float(np.dot(rel_pos, rel_vel)) / rel_speed_sq
    is_converging = tca > 0.0

    if is_converging:
        dca_vec = rel_pos + rel_vel * tca
        dca = float(np.linalg.norm(dca_vec))
    else:
        dca = current_dist

    return TrajectoryConvergence(
        track_id_1=0,
        track_id_2=0,
        distance_m=current_dist,
        is_converging=is_converging,
        tca_sec=tca if is_converging else float("inf"),
        dca_m=dca,
    )


def calculate_directional_cosine(
    heading_vec: Tuple[float, float],
    lane_vec: Tuple[float, float],
) -> float:
    """Calculate cosine of angle between vehicle heading and permitted lane direction.
    cos = (u_h . u_l) / (||u_h|| * ||u_l||)
    """
    uh = np.array(heading_vec, dtype=np.float64)
    ul = np.array(lane_vec, dtype=np.float64)
    norm_h = np.linalg.norm(uh)
    norm_l = np.linalg.norm(ul)
    if norm_h < 1e-4 or norm_l < 1e-4:
        return 1.0  # Cannot determine direction; default neutral
    return float(np.dot(uh, ul) / (norm_h * norm_l))


def compute_pixel_kinematics(tr: Any, pts_ns: int) -> Tuple[float, float]:
    """Compute smoothed pixel speed (px/s) and signed pixel acceleration (px/s^2) from track history."""
    if hasattr(tr, "timestamps") and hasattr(tr, "position_history"):
        if pts_ns in tr.timestamps:
            idx = tr.timestamps.index(pts_ns)
            pos_hist = tr.position_history[:idx + 1]
            time_hist = tr.timestamps[:idx + 1]
        else:
            pos_hist = tr.position_history
            time_hist = tr.timestamps

        n_pts = len(pos_hist)
        if n_pts >= 2 and len(time_hist) >= 2:
            dt = (time_hist[-1] - time_hist[-2]) * 1e-9
            if dt < 1e-5:
                dt = 1.0 / 30.0

            # Windowed linear regression for velocity over up to 4 recent points
            k = min(4, n_pts)
            w_pos = pos_hist[-k:]
            w_time = [(t - time_hist[-1]) * 1e-9 for t in time_hist[-k:]]

            if k >= 3 and (w_time[-1] - w_time[0]) > 1e-4:
                t_arr = np.array(w_time, dtype=np.float64)
                x_arr = np.array([p[0] for p in w_pos], dtype=np.float64)
                y_arr = np.array([p[1] for p in w_pos], dtype=np.float64)
                t_mean = np.mean(t_arr)
                t_var = np.sum((t_arr - t_mean) ** 2)
                if t_var > 1e-6:
                    vx_px = float(np.sum((t_arr - t_mean) * (x_arr - np.mean(x_arr))) / t_var)
                    vy_px = float(np.sum((t_arr - t_mean) * (y_arr - np.mean(y_arr))) / t_var)
                    px_spd = float(np.hypot(vx_px, vy_px))
                else:
                    dp = float(np.hypot(pos_hist[-1][0] - pos_hist[-2][0], pos_hist[-1][1] - pos_hist[-2][1]))
                    px_spd = dp / dt
            else:
                dp = float(np.hypot(pos_hist[-1][0] - pos_hist[-2][0], pos_hist[-1][1] - pos_hist[-2][1]))
                px_spd = dp / dt

            # Compute acceleration from prior speed history if available
            px_acc = 0.0
            if n_pts >= 3 and len(time_hist) >= 3:
                dt_prev = (time_hist[-2] - time_hist[-3]) * 1e-9
                if dt_prev < 1e-5:
                    dt_prev = 1.0 / 30.0
                dp_prev = float(np.hypot(pos_hist[-2][0] - pos_hist[-3][0], pos_hist[-2][1] - pos_hist[-3][1]))
                px_spd_prev = dp_prev / dt_prev
                # Signed acceleration: negative if slowing down
                raw_px_acc = (px_spd - px_spd_prev) / dt
                # Clamp physically: impact decel up to -500 px/s^2, but positive acceleration
                # is traction-bounded (max ~80 px/s^2 for vehicles) to eliminate forward flicker spikes
                px_acc = float(np.clip(raw_px_acc, -500.0, 80.0))
                if px_spd < 2.0 and px_spd_prev < 2.0:
                    px_acc = 0.0  # Stationary noise deadband
            return px_spd, px_acc
    return 0.0, 0.0

