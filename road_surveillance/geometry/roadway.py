"""Roadway Boundary Geometry & Evidence-Gated Shoulder Analysis Module.

Iteration R: Daylight Intersection Hardening & Trajectory-Gated Shoulder Recovery.
Provides:
- Exact distance to polygon boundary and nearest edge calculation
- Boundary segment outward normal vector derivation
- Evidence-gated shoulder corridor admission for solitary vehicle barrier strikes:
  1. Solitary track with no interacting lead vehicle.
  2. Vehicle ground contact point lies within 0.8 meters of road polygon outer boundary.
  3. Outward velocity vector pointing toward the shoulder/barrier (v · n_boundary > 0).
  4. Active deceleration or trajectory deflection (Δa <= -3.0 m/s^2 or Δyaw >= 20°).
  Allows detection/tracking admission up to +1.2 meters beyond the boundary along projection ray.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


def point_to_segment_distance(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Calculate Euclidean distance from 2D point p to line segment ab.

    Returns:
        Tuple of (distance, closest_point_on_segment, projection_parameter_t)
    """
    ab = b - a
    norm_ab_sq = float(np.dot(ab, ab))
    if norm_ab_sq < 1e-8:
        dist = float(np.linalg.norm(p - a))
        return dist, a.copy(), 0.0

    t = float(np.clip(np.dot(p - a, ab) / norm_ab_sq, 0.0, 1.0))
    projection = a + t * ab
    dist = float(np.linalg.norm(p - projection))
    return dist, projection, t


def compute_polygon_centroid(polygon: List[Tuple[float, float]]) -> np.ndarray:
    """Computes geometric centroid of polygon."""
    if not polygon:
        return np.array([0.0, 0.0], dtype=np.float64)
    arr = np.array(polygon, dtype=np.float64)
    return np.mean(arr, axis=0)


def find_nearest_boundary_segment(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> Tuple[float, Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """Find the nearest polygon boundary segment to a 2D point.

    Returns:
        min_dist: Distance in coordinates to closest edge
        segment: (a, b) endpoints of closest segment
        closest_pt: Projection point on segment
        outward_normal: Unit vector perpendicular to segment pointing outward from polygon
    """
    p_arr = np.array([point[0], point[1]], dtype=np.float64)
    n = len(polygon)
    if n < 2:
        return 99.0, (p_arr, p_arr), p_arr, np.array([0.0, 1.0], dtype=np.float64)

    centroid = compute_polygon_centroid(polygon)

    min_dist = float("inf")
    best_seg = (np.array(polygon[0]), np.array(polygon[1]))
    best_proj = p_arr.copy()

    for i in range(n):
        a = np.array(polygon[i], dtype=np.float64)
        b = np.array(polygon[(i + 1) % n], dtype=np.float64)
        dist, proj, _ = point_to_segment_distance(p_arr, a, b)
        if dist < min_dist:
            min_dist = dist
            best_seg = (a, b)
            best_proj = proj

    # Compute outward normal
    a, b = best_seg
    ab = b - a
    seg_len = float(np.linalg.norm(ab))
    if seg_len < 1e-6:
        outward_normal = np.array([0.0, 1.0], dtype=np.float64)
    else:
        # Candidate normals
        n1 = np.array([-ab[1], ab[0]], dtype=np.float64) / seg_len
        n2 = -n1
        # Outward normal points away from polygon centroid
        mid = (a + b) / 2.0
        to_mid = mid - centroid
        if np.dot(n1, to_mid) >= np.dot(n2, to_mid):
            outward_normal = n1
        else:
            outward_normal = n2

    return min_dist, best_seg, best_proj, outward_normal


def is_point_inside_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]],
) -> bool:
    """Ray casting point-in-polygon test."""
    if not polygon or len(polygon) < 3:
        return True
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def check_evidence_gated_shoulder_allowance(
    ground_pos: Tuple[float, float],
    velocity_vector: Tuple[float, float],
    accel_ms2: float,
    yaw_delta_deg: float,
    has_lead_vehicle: bool,
    roadway_polygon: List[Tuple[float, float]],
    homography: Optional[Any] = None,
    pixel_scale_m_per_px: float = 0.02,
    max_boundary_dist_m: float = 0.8,
    max_extension_m: float = 1.2,
    current_speed_kmh: float = 0.0,
    max_recent_speed_kmh: float = 0.0,
    min_v_dot_n: float = 0.35,
    min_impact_accel_ms2: float = 8.0,
    max_post_speed_kmh: float = 15.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluates the Task R.2 / S.A.4 Evidence-Gated Shoulder Corridor Search.

    Task S.A.4 Shoulder barrier protection:
    Preserve R.2 shoulder recovery (e.g. UarP8qU1S-c_01), while requiring genuine barrier-strike
    evidence beyond boundary proximity:
    1. Solitary track with no interacting lead vehicle.
    2. Vehicle ground contact point lies within 0.8 meters of road polygon outer boundary.
    3. Directional outward velocity pointing toward boundary normal (v · n_boundary >= 0.35).
    4. Acceleration impulse: deceleration <= -8.0 m/s^2 or |accel| >= 12.0 m/s^2.
    5. Post-event velocity or arrest: speed <= 15.0 km/h or speed drop >= 10.0 km/h or yaw >= 25 deg.

    Action:
    Permits acceptance up to +1.2 meters beyond the roadway boundary exclusively along
    the vehicle's directional projection ray. Discards cruising boundary vehicles.

    Returns:
        Tuple of (is_admitted, audit_telemetry)
    """
    speed_drop = max(0.0, max_recent_speed_kmh - current_speed_kmh)
    telemetry: Dict[str, Any] = {
        "condition_solitary": not has_lead_vehicle,
        "condition_boundary_proximity": False,
        "condition_outward_velocity": False,
        "condition_normal_alignment": False,
        "condition_acceleration_impulse": False,
        "condition_velocity_arrest": False,
        "dist_to_boundary_m": 99.0,
        "v_dot_n": 0.0,
        "accel_ms2": round(accel_ms2, 2),
        "curr_spd_kmh": round(current_speed_kmh, 2),
        "max_recent_spd_kmh": round(max_recent_speed_kmh, 2),
        "speed_drop_kmh": round(speed_drop, 2),
        "yaw_delta_deg": round(yaw_delta_deg, 2),
        "extension_m": 0.0,
        "is_admitted": False,
    }

    if not roadway_polygon or len(roadway_polygon) < 3:
        return False, telemetry

    # Coordinate alignment: check if polygon is in pixels or metric ground
    max_poly = max(max(abs(v[0]), abs(v[1])) for v in roadway_polygon)
    max_pt = max(abs(ground_pos[0]), abs(ground_pos[1]))

    poly_eval = roadway_polygon
    pt_eval = ground_pos

    if max_poly > 150.0 and max_pt <= 150.0 and homography is not None:
        try:
            poly_ground = []
            for px, py in roadway_polygon:
                gx, gy = homography.pixel_to_ground(px, py)
                poly_ground.append((gx, gy))
            poly_eval = poly_ground
        except Exception:
            poly_eval = roadway_polygon

    is_pixel_eval = max(max(abs(v[0]), abs(v[1])) for v in poly_eval) > 150.0
    scale = pixel_scale_m_per_px if is_pixel_eval else 1.0

    dist_raw, best_seg, best_proj, n_outward = find_nearest_boundary_segment(pt_eval, poly_eval)
    dist_m = float(dist_raw * scale)
    telemetry["dist_to_boundary_m"] = round(dist_m, 3)

    is_inside = is_point_inside_polygon(pt_eval, poly_eval)

    # Within 0.8 meters of the road polygon outer boundary
    if is_inside:
        boundary_proximity = (dist_m <= max_boundary_dist_m)
        extension_dist_m = 0.0
    else:
        boundary_proximity = (dist_m <= max_extension_m)
        extension_dist_m = dist_m

    telemetry["condition_boundary_proximity"] = boundary_proximity
    telemetry["extension_m"] = round(extension_dist_m, 3)

    # Outward velocity vector pointing toward the shoulder/barrier normal
    vx, vy = velocity_vector
    v_norm = float(np.hypot(vx, vy))
    if v_norm > 1e-4:
        u_v = np.array([vx, vy], dtype=np.float64) / v_norm
        v_dot_n = float(np.dot(u_v, n_outward))
    else:
        v_dot_n = 0.0

    telemetry["v_dot_n"] = round(v_dot_n, 3)
    has_outward_v = (v_dot_n > 0.05)
    normal_alignment = (v_dot_n >= min_v_dot_n)
    telemetry["condition_outward_velocity"] = has_outward_v
    telemetry["condition_normal_alignment"] = normal_alignment

    # 1. Solitary track check
    if has_lead_vehicle:
        return False, telemetry

    if not boundary_proximity:
        return False, telemetry

    if not normal_alignment:
        return False, telemetry

    # 4. Acceleration impulse check (a <= -8.0 or |a| >= 12.0)
    accel_impulse = (accel_ms2 <= -min_impact_accel_ms2 or abs(accel_ms2) >= 12.0)
    telemetry["condition_acceleration_impulse"] = accel_impulse
    if not accel_impulse:
        return False, telemetry

    # 5. Post-event velocity or arrest check
    vel_arrest = (
        current_speed_kmh <= max_post_speed_kmh
        or speed_drop >= 10.0
        or yaw_delta_deg >= 25.0
    )
    telemetry["condition_velocity_arrest"] = vel_arrest
    if not vel_arrest:
        return False, telemetry

    # Action: All conditions hold! Permit acceptance up to +1.2m along directional ray
    telemetry["is_admitted"] = True
    return True, telemetry

