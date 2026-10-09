"""Single-Vehicle Kinematic Recovery & Hypothesis Arbitration Module.

Iteration M: Single-Vehicle Kinematic Recovery & Hypothesis Arbitration
Objective:
Recover single-vehicle incidents (spin-outs, barrier strikes, rollovers, run-off-road)
without hard threshold failure, using continuous composite evidence scoring:
S_single = (0.35 * S_theta) + (0.35 * S_decel) + (0.15 * S_bound) + (0.15 * S_ar)
Candidate trigger: S_single >= tau_single (tau_single in [0.55, 0.70])
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from road_surveillance.tracking.track_state import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics
from road_surveillance.calibration.homography import PlanarHomography


def calculate_heading_deviation(positions: List[Tuple[float, float]], k: int = 5) -> float:
    """Calculate trajectory angular deviation in degrees over a window of k points.

    Computes the cosine distance between the past heading vector and the current heading vector:
    Vector past: from positions[-k-1] to positions[-mid-1]
    Vector curr: from positions[-mid-1] to positions[-1]

    Parameters:
        positions: List of (x, y) coordinates in chronological order.
        k: Window size for heading comparison (default 5).

    Returns:
        float: Angular deviation in degrees [0.0, 180.0].
    """
    if not positions or len(positions) < 3:
        return 0.0

    actual_k = min(k, len(positions) - 1)
    if actual_k < 2:
        return 0.0

    mid = max(1, actual_k // 2)
    p_start = np.array(positions[-actual_k - 1], dtype=np.float64)
    p_mid = np.array(positions[-mid - 1], dtype=np.float64)
    p_curr = np.array(positions[-1], dtype=np.float64)

    v_past = p_mid - p_start
    v_curr = p_curr - p_mid

    norm_past = float(np.linalg.norm(v_past))
    norm_curr = float(np.linalg.norm(v_curr))

    # Reject tiny displacements / stationary jitter
    is_pixel = max(abs(p_curr[0]), abs(p_curr[1])) > 150.0
    min_disp = 5.0 if is_pixel else 0.5
    if norm_past < min_disp or norm_curr < min_disp:
        return 0.0

    cos_val = np.dot(v_past, v_curr) / (norm_past * norm_curr)
    cos_val = float(np.clip(cos_val, -1.0, 1.0))

    theta_deg = float(np.degrees(np.arccos(cos_val)))
    return theta_deg


def point_to_segment_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Calculate Euclidean distance from 2D point p to line segment ab."""
    ab = b - a
    norm_ab_sq = float(np.dot(ab, ab))
    if norm_ab_sq < 1e-8:
        return float(np.linalg.norm(p - a))

    t = np.clip(np.dot(p - a, ab) / norm_ab_sq, 0.0, 1.0)
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def distance_to_polygon_boundary(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
    homography: Optional[PlanarHomography] = None,
    pixel_scale_m_per_px: float = 0.02,
) -> float:
    """Calculate minimum distance in meters from vehicle ground anchor to roadway edge polygon.

    Parameters:
        point: Vehicle ground anchor (x, y) - can be ground plane (m) or pixel coords.
        polygon: Roadway edge polygon vertices [(x, y), ...].
        homography: Optional homography for coordinate transformations.
        pixel_scale_m_per_px: Fallback scale when operating purely in pixel space.

    Returns:
        float: Estimated distance to roadway edge in meters.
    """
    if not polygon or len(polygon) < 2:
        return 5.0  # Safe default: vehicle considered far from edge

    pt_x, pt_y = point
    p_arr = np.array([pt_x, pt_y], dtype=np.float64)

    max_poly = max(max(abs(v[0]), abs(v[1])) for v in polygon)
    max_pt = max(abs(pt_x), abs(pt_y))

    # Case A: Polygon is in pixel coordinates (> 150) and point is in ground coordinates (<= 150)
    if max_poly > 150.0 and max_pt <= 150.0 and homography is not None:
        try:
            poly_ground = []
            for px, py in polygon:
                gx, gy = homography.pixel_to_ground(px, py)
                poly_ground.append(np.array([gx, gy], dtype=np.float64))
            min_dist = float("inf")
            n = len(poly_ground)
            for i in range(n):
                d = point_to_segment_distance(p_arr, poly_ground[i], poly_ground[(i + 1) % n])
                if d < min_dist:
                    min_dist = d
            return float(min_dist)
        except Exception:
            pass

    # Case B: Both in pixel coordinates
    if max_poly > 150.0 and max_pt > 150.0:
        min_dist_px = float("inf")
        n = len(polygon)
        for i in range(n):
            a = np.array(polygon[i], dtype=np.float64)
            b = np.array(polygon[(i + 1) % n], dtype=np.float64)
            d = point_to_segment_distance(p_arr, a, b)
            if d < min_dist_px:
                min_dist_px = d
        return float(min_dist_px * pixel_scale_m_per_px)

    # Case C: Both in metric ground coordinates
    min_dist_m = float("inf")
    n = len(polygon)
    for i in range(n):
        a = np.array(polygon[i], dtype=np.float64)
        b = np.array(polygon[(i + 1) % n], dtype=np.float64)
        d = point_to_segment_distance(p_arr, a, b)
        if d < min_dist_m:
            min_dist_m = d

    return float(min_dist_m)


def check_has_lead_vehicle(
    track_id: int,
    tracks: Dict[int, Any],
    kinematics: Optional[Dict[int, Any]] = None,
    max_forward_dist_m: float = 20.0,
    max_lateral_dist_m: float = 3.0,
) -> bool:
    """Evaluates whether an immediate lead vehicle is present in the forward corridor."""
    subject_track = tracks.get(track_id)
    if not subject_track:
        return False

    sub_kin = kinematics.get(track_id) if kinematics else None
    if sub_kin is not None:
        heading_vec = getattr(sub_kin, "heading_vector", (0.0, 1.0))
        sub_pos = getattr(sub_kin, "ground_pos", (0.0, 0.0))
    elif subject_track.ground_history:
        sub_pos = subject_track.ground_history[-1]
        heading_vec = (0.0, 1.0)
    else:
        return False

    uh = np.array(heading_vec, dtype=np.float64)
    norm_uh = np.linalg.norm(uh)
    if norm_uh < 1e-4:
        uh = np.array([0.0, 1.0], dtype=np.float64)
    else:
        uh = uh / norm_uh

    # Perpendicular vector (lateral direction)
    u_lat = np.array([-uh[1], uh[0]], dtype=np.float64)

    for other_tid, other_track in tracks.items():
        if other_tid == track_id:
            continue
        if getattr(other_track, "class_label", "vehicle").lower() not in Track.VEHICLE_CLASSES:
            continue

        other_kin = kinematics.get(other_tid) if kinematics else None
        if other_kin is not None:
            other_pos = getattr(other_kin, "ground_pos", (0.0, 0.0))
        elif other_track.ground_history:
            other_pos = other_track.ground_history[-1]
        else:
            continue

        rel_vec = np.array(other_pos, dtype=np.float64) - np.array(sub_pos, dtype=np.float64)
        long_dist = float(np.dot(rel_vec, uh))
        lat_dist = float(abs(np.dot(rel_vec, u_lat)))

        # Check forward corridor: in front (0 < long_dist <= max_forward_dist_m) and within lane
        if 0.5 <= long_dist <= max_forward_dist_m and lat_dist <= max_lateral_dist_m:
            return True

    return False


def compute_single_vehicle_evidence(
    track: Any,
    roadway_poly: Optional[List[Tuple[float, float]]] = None,
    homography: Optional[PlanarHomography] = None,
    window_k: int = 5,
    kinematics: Optional[Any] = None,
    has_lead_vehicle: Optional[bool] = None,
    weights: Tuple[float, float, float, float] = (0.35, 0.35, 0.15, 0.15),
    heading_deviation_deg: Optional[float] = None,
    aspect_ratio_delta: Optional[float] = None,
) -> Tuple[float, Dict[str, float]]:
    """Calculates continuous composite evidence score S_single for a single vehicle.

    Formula:
        S_single = (w_theta * S_theta) + (w_decel * S_decel) + (w_bound * S_bound) + (w_ar * S_ar)
        Weights: w_theta=0.35, w_decel=0.35, w_bound=0.15, w_ar=0.15

    Sub-scores:
        1. S_theta: Trajectory Angular Deviation Score
           theta_deg = calculate_heading_deviation(positions, k=window_k)
           S_theta = np.clip((theta_deg - 15.0) / (45.0 - 15.0), 0.0, 1.0)
        2. S_decel: Solitary Kinetic Energy / Deceleration Dissipation
           decel_mag = -longitudinal_accel (positive when braking)
           S_decel = np.clip((decel_mag - 2.5) / (6.0 - 2.5), 0.0, 1.0) if not has_lead_vehicle else 0.0
        3. S_bound: Roadway Boundary Proximity Bias
           d_edge_m = distance_to_polygon_boundary(ground_anchor, roadway_poly)
           S_bound = np.clip((2.5 - d_edge_m) / (2.5 - 0.5), 0.0, 1.0)
        4. S_ar: Aspect Ratio Distortion Rate
           delta_ar = abs(current_ar - smoothed_ar)
           S_ar = np.clip((delta_ar - 0.15) / (0.40 - 0.15), 0.0, 1.0)

    Returns:
        Tuple[float, Dict[str, float]]:
            s_single: Composite evidence score [0.0, 1.0]
            telemetry: Detailed breakdown of continuous sub-scores and raw measurements
    """
    w_theta, w_decel, w_bound, w_ar = weights

    # --- 1. Trajectory Angular Deviation (S_theta) ---
    positions = getattr(track, "positions", getattr(track, "position_history", []))
    if not positions and hasattr(track, "ground_history") and track.ground_history:
        positions = track.ground_history

    theta_deg = calculate_heading_deviation(positions, k=window_k)
    if heading_deviation_deg is not None:
        theta_deg = max(theta_deg, float(heading_deviation_deg))
    s_theta = float(np.clip((theta_deg - 15.0) / (45.0 - 15.0), 0.0, 1.0))

    # --- 2. Deceleration Dissipation (S_decel) ---
    # Retrieve longitudinal acceleration (signed: negative when slowing down)
    long_accel = 0.0
    if hasattr(track, "longitudinal_accel"):
        long_accel = float(track.longitudinal_accel)
    elif kinematics is not None:
        is_calib = getattr(kinematics, "is_calibrated", False)
        acc_mag = float(getattr(kinematics, "acceleration_mag", 0.0))
        px_acc = float(getattr(kinematics, "pixel_acceleration", 0.0))
        if is_calib and abs(acc_mag) > 0.1:
            long_accel = acc_mag
        elif px_acc != 0.0 and hasattr(track, "current_bbox_xywh"):
            w_box, h_box = track.current_bbox_xywh[2], track.current_bbox_xywh[3]
            char_dim = max(15.0, max(w_box, h_box))
            body_acc = (px_acc / char_dim) * 4.5
            long_accel = body_acc if abs(body_acc) > abs(acc_mag) else acc_mag
        else:
            long_accel = acc_mag
    elif hasattr(track, "acceleration_mag"):
        long_accel = float(track.acceleration_mag)

    decel_mag = -long_accel  # positive when braking
    decel_mag = float(np.clip(decel_mag, -60.0, 60.0))

    lead_veh_present = False
    if has_lead_vehicle is not None:
        lead_veh_present = has_lead_vehicle
    elif hasattr(track, "has_lead_vehicle"):
        lead_veh_present = bool(track.has_lead_vehicle)

    if lead_veh_present:
        s_decel = 0.0
    else:
        s_decel = float(np.clip((decel_mag - 2.5) / (6.0 - 2.5), 0.0, 1.0))

    # --- 3. Roadway Boundary Proximity (S_bound) ---
    ground_pos = None
    if hasattr(track, "ground_xy"):
        ground_pos = track.ground_xy
    elif hasattr(track, "ground_pos"):
        ground_pos = track.ground_pos
    elif kinematics is not None and hasattr(kinematics, "ground_pos"):
        ground_pos = kinematics.ground_pos
    elif hasattr(track, "ground_history") and track.ground_history:
        ground_pos = track.ground_history[-1]
    elif positions:
        ground_pos = positions[-1]
    else:
        ground_pos = (0.0, 0.0)

    shoulder_admitted = False
    shoulder_telemetry = {}
    if roadway_poly and kinematics is not None:
        try:
            from road_surveillance.geometry.roadway import check_evidence_gated_shoulder_allowance
            v_vec = getattr(kinematics, "velocity_vector", (0.0, 0.0))
            curr_spd_val = float(getattr(kinematics, "speed_kmh", getattr(track, "speed_kmh", 0.0)))
            spd_hist = getattr(track, "speed_history", [])
            max_spd_val = max(spd_hist) if spd_hist else curr_spd_val

            is_sh, sh_tel = check_evidence_gated_shoulder_allowance(
                ground_pos=ground_pos,
                velocity_vector=v_vec,
                accel_ms2=long_accel,
                yaw_delta_deg=theta_deg,
                has_lead_vehicle=lead_veh_present,
                roadway_polygon=roadway_poly,
                homography=homography,
                current_speed_kmh=curr_spd_val,
                max_recent_speed_kmh=max_spd_val,
            )
            shoulder_admitted = is_sh
            shoulder_telemetry = sh_tel
        except Exception:
            pass

    if roadway_poly:
        d_edge_m = distance_to_polygon_boundary(ground_pos, roadway_poly, homography)
        if shoulder_admitted:
            d_edge_m = min(d_edge_m, 0.5)
            s_bound = 1.0
        else:
            s_bound = float(np.clip((2.5 - d_edge_m) / (2.5 - 0.5), 0.0, 1.0))
    else:
        d_edge_m = 1.5
        s_bound = 0.50

    # --- 4. Aspect Ratio Distortion (S_ar) ---
    current_ar = 1.0
    smoothed_ar = 1.0

    if hasattr(track, "current_ar") and hasattr(track, "smoothed_ar"):
        current_ar = float(track.current_ar)
        smoothed_ar = float(track.smoothed_ar)
    elif hasattr(track, "aspect_ratio") and hasattr(track, "smoothed_aspect_ratio"):
        current_ar = float(track.aspect_ratio)
        smoothed_ar = float(track.smoothed_aspect_ratio)
    elif hasattr(track, "bbox_history") and track.bbox_history:
        curr_box = track.current_bbox_xywh if hasattr(track, "current_bbox_xywh") else track.bbox_history[-1]
        w_box, h_box = curr_box[2], curr_box[3]
        current_ar = float(w_box / max(1.0, h_box))

        recent_boxes = track.bbox_history[-10:]
        ar_hist = [float(b[2] / max(1.0, b[3])) for b in recent_boxes]
        smoothed_ar = float(np.median(ar_hist))
    else:
        current_ar = 1.0
        smoothed_ar = 1.0

    delta_ar = abs(current_ar - smoothed_ar)
    if aspect_ratio_delta is not None:
        delta_ar = max(delta_ar, float(aspect_ratio_delta))
    s_ar = float(np.clip((delta_ar - 0.15) / (0.40 - 0.15), 0.0, 1.0))

    # --- Composite Score ---
    s_single = float((w_theta * s_theta) + (w_decel * s_decel) + (w_bound * s_bound) + (w_ar * s_ar))

    breakdown = {
        "s_single": round(s_single, 4),
        "s_theta": round(s_theta, 4),
        "s_decel": round(s_decel, 4),
        "s_bound": round(s_bound, 4),
        "s_ar": round(s_ar, 4),
        "theta_deg": round(theta_deg, 2),
        "decel_mag": round(decel_mag, 2),
        "d_edge_m": round(d_edge_m, 2),
        "delta_ar": round(delta_ar, 4),
        "has_lead_vehicle": lead_veh_present,
        "shoulder_admitted": shoulder_admitted,
        "shoulder_telemetry": shoulder_telemetry,
        "v_dot_n": round(float(shoulder_telemetry.get("v_dot_n", 0.0)), 3),
    }

    return s_single, breakdown
