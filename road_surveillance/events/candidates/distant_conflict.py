"""Distant Trajectory Conflict Incident Candidate Detector (Iteration D).

Detects 2-vehicle collision candidates in high-perspective or distant camera views
where 2D bounding boxes do not achieve visual overlap:
Track A ────────→
                  \
                   → predicted trajectory conflict (TCA / DCA)
                  /
Track B ────────→
        ↓
Heading convergence + Relative velocity + Time-to-conflict + Spatial proximity
        ↓
Distant Collision Candidate
        ↓
Incident Lifecycle Management
"""

import math
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager


class DistantTrajectoryConflictDetector(CandidateDetector):
    """Detects distant non-overlapping trajectory conflicts between vehicle pairs."""

    VEHICLE_LABELS = {"car", "truck", "bus", "van", "motorcycle", "train", "vehicle", "candidate_vehicle_recovered"}

    def __init__(
        self,
        is_camera_calibrated: bool = False,
        video_warmup_frames: int = 15,
        min_track_age: int = 8,
        deceleration_threshold_ms2: float = -3.5,
        max_physical_decel_ms2: float = 60.0,
        tca_min_sec: float = 0.05,
        tca_max_sec: float = 2.0,
        dca_norm_max: float = 1.2,
        dca_metric_max_m: float = 3.5,
        dist_norm_max: float = 3.5,
        dist_metric_max_m: float = 18.0,
        min_closing_speed_kmh: float = 12.0,
        min_persistence_frames: int = 2,
    ):
        self.is_camera_calibrated = is_camera_calibrated
        self.video_warmup_frames = video_warmup_frames
        self.min_track_age = min_track_age
        self.decel_threshold = deceleration_threshold_ms2
        self.max_physical_decel = max_physical_decel_ms2
        self.tca_min = tca_min_sec
        self.tca_max = tca_max_sec
        self.dca_norm_max = dca_norm_max
        self.dca_metric_max = dca_metric_max_m
        self.dist_norm_max = dist_norm_max
        self.dist_metric_max = dist_metric_max_m
        self.min_closing_spd = min_closing_speed_kmh
        self.min_persistence = min_persistence_frames
        self.wrong_way_handler = None
        self.suppressed_events: List[Dict[str, Any]] = []

        self.spatial_clusters: List[Dict[str, Any]] = []
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_lateral_distances: Dict[Tuple[int, int], float] = {}
        self.prev_lateral_dists: Dict[Tuple[int, int], List[float]] = {}
        self.prev_headings: Dict[int, List[float]] = {}

    def log_suppressed(self, frame_idx: int, tracks: List[int], reason: str, dist_or_dca: float) -> None:
        val = round(float(dist_or_dca), 2)
        self.suppressed_events.append({
            "frame": frame_idx,
            "tracks": list(tracks),
            "track_ids": list(tracks),
            "module": "distant_trajectory_conflict",
            "reason": reason,
            "distance_or_dca": val,
            "metric_val": val,
        })

    def reset(self) -> None:
        """Reset internal state between videos."""
        self.spatial_clusters.clear()
        self.prev_distances.clear()
        self.prev_lateral_distances.clear()
        self.prev_lateral_dists.clear()
        self.prev_headings.clear()
        self.suppressed_events.clear()

    def _estimate_metric_acceleration(self, bbox_xywh: Tuple[float, float, float, float], raw_accel: float) -> float:
        """Estimate metric acceleration in m/s^2."""
        if self.is_camera_calibrated:
            return raw_accel
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

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
        candidates: List[EventCandidate] = []
        current_pair_keys: Set[Tuple[int, int]] = set()

        if frame_idx < self.video_warmup_frames:
            return candidates

        for conv in convergences:
            t1_id = conv.track_id_1
            t2_id = conv.track_id_2
            k1 = kinematics.get(t1_id)
            k2 = kinematics.get(t2_id)
            tr1 = tracks.get(t1_id)
            tr2 = tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            if tr1.class_label.lower() not in self.VEHICLE_LABELS or tr2.class_label.lower() not in self.VEHICLE_LABELS:
                continue

            # 1. Track Maturity Gate
            if tr1.track_age < self.min_track_age or tr2.track_age < self.min_track_age:
                continue

            # 2. Kinematic Warm-Up Gate
            if len(tr1.ground_history) < 3 or len(tr2.ground_history) < 3:
                continue

            # Re-acquisition Settlement Gate: tracks recovering from coasting/lost must have >= 3 matched frames
            if (not getattr(tr1, "is_reacquisition_settled", True)) or (not getattr(tr2, "is_reacquisition_settled", True)):
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)
            dist = conv.distance_m

            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh

            # Edge margin clipping
            if b1[0] < 15.0 or b1[1] < 15.0 or b2[0] < 15.0 or b2[1] < 15.0:
                continue

            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
            px_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_dist = (dist / char_dim) if self.is_camera_calibrated else (px_dist / char_dim)

            # =========================================================================
            # TWO-DOMAIN TRAJECTORY CONFLICT FORMULATION (CANDIDATE B)
            # =========================================================================

            # Step 0: Kinetic Prerequisite
            # Both participating tracks must be active dynamic vehicles
            v1_kmh = float(k1.speed_kmh)
            v2_kmh = float(k2.speed_kmh)
            if min(v1_kmh, v2_kmh) < 10.0:
                self.log_suppressed(frame_idx, [t1_id, t2_id], "EXCLUDED_STATIONARY", min(v1_kmh, v2_kmh))
                continue

            # Multi-Frame Feature Extraction (over continuous window [t - 4, t])
            # Retrieve 5-frame ground positions: track.ground_history stores (smooth_X, smooth_Y) per frame
            # Current frame ground position is k.ground_pos; previous frames are in track.ground_history
            p1_curr = k1.ground_pos
            p2_curr = k2.ground_pos
            hist1 = tr1.ground_history
            hist2 = tr2.ground_history

            # Need at least 4 preceding points to form 5-point window [t-4, t-3, t-2, t-1, t]
            if len(hist1) < 4 or len(hist2) < 4:
                self.log_suppressed(frame_idx, [t1_id, t2_id], "SUPPRESSED_INSUFFICIENT_HISTORY", float(len(hist1)))
                continue

            # p(t-4) is 4 frames back from current
            p1_t4 = hist1[-4]
            p2_t4 = hist2[-4]

            # Timing delta for 4 frames:
            if len(tr1.timestamps) >= 5 and len(tr2.timestamps) >= 5:
                dt_total = float((tr1.timestamps[-1] - tr1.timestamps[-5]) * 1e-9)
            else:
                dt_total = 4.0 / 30.0
            if dt_total <= 1e-4:
                dt_total = 4.0 / 30.0

            # 5-frame endpoint velocity vectors:
            v1_bar = np.array([(p1_curr[0] - p1_t4[0]) / dt_total, (p1_curr[1] - p1_t4[1]) / dt_total], dtype=np.float64)
            v2_bar = np.array([(p2_curr[0] - p2_t4[0]) / dt_total, (p2_curr[1] - p2_t4[1]) / dt_total], dtype=np.float64)

            spd1_bar = float(np.linalg.norm(v1_bar))
            spd2_bar = float(np.linalg.norm(v2_bar))

            dot_v = float(np.dot(v1_bar, v2_bar))
            denom_v = (spd1_bar * spd2_bar) + 1e-6
            cos_v_clamped = max(-1.0, min(1.0, dot_v / denom_v))
            delta_theta_5f = float(np.degrees(np.arccos(cos_v_clamped)))

            # Lateral distance and 5-frame displacement:
            d_lat_t4 = abs(p1_t4[0] - p2_t4[0])
            d_lat = abs(p1_curr[0] - p2_curr[0])
            delta_d_lat_5f = d_lat - d_lat_t4

            # 5-frame linear extrapolation (TCA & DCA):
            pos1_t = np.array(p1_curr, dtype=np.float64)
            pos2_t = np.array(p2_curr, dtype=np.float64)
            r_rel = pos2_t - pos1_t
            v_rel = v2_bar - v1_bar
            dist_curr = float(np.linalg.norm(r_rel))
            v_rel_sq = float(np.dot(v_rel, v_rel))

            if v_rel_sq > 1e-4:
                tca_5f = -float(np.dot(r_rel, v_rel)) / v_rel_sq
                if tca_5f > 0.0:
                    dca_vec = r_rel + v_rel * tca_5f
                    dca_5f = float(np.linalg.norm(dca_vec))
                else:
                    dca_5f = dist_curr
            else:
                tca_5f = float("inf")
                dca_5f = dist_curr

            # 5-frame relative corridor velocity (km/h)
            v_rel_corridor_kmh = float(np.linalg.norm(v_rel) * 3.6)

            # Step 1: Domain 1 — Active Approaching Encroachment
            is_domain_1 = bool(
                (1.80 <= d_lat <= 3.80)
                and (delta_theta_5f >= 12.0)
                and (delta_d_lat_5f <= -0.15)
                and (dca_5f <= 1.00)
                and (tca_5f <= 2.50)
            )

            # Step 2: Domain 2 — Candidate Physical Conflict Gate
            # Base Eligibility: Ground history >= 5 frames, min(v1, v2) >= 10 km/h, shared lateral corridor d_lat < 1.80m
            is_domain_2 = False
            domain_2_path = None
            if d_lat < 1.80:
                # Longitudinal separation (d_long)
                d_long = abs(p1_curr[1] - p2_curr[1])

                # Relative speed v_rel (corridor relative speed in m/s and km/h)
                v_rel_mps = float(np.linalg.norm(v_rel))
                v_rel_kmh = v_rel_mps * 3.6

                # Frame step time delta dt
                if len(tr1.timestamps) >= 2:
                    dt_frame = float((tr1.timestamps[-1] - tr1.timestamps[-2]) * 1e-9)
                else:
                    dt_frame = 1.0 / 30.0
                if dt_frame <= 1e-4:
                    dt_frame = 1.0 / 30.0

                # -------------------------------------------------------------
                # Candidate Path A (Active Lateral Encroachment / Cut-In):
                # - Sustained lateral approach velocity: |d_dot_lat| >= 0.40 m/s across >= 3 consecutive frames
                # - Longitudinal headway: TTC_long = d_long / v_rel <= 1.50 s (where v_rel >= 12.0 km/h)
                # -------------------------------------------------------------
                path_a = False
                lat_hist = self.prev_lateral_dists.get(pair_key, [])
                lat_hist.append(float(d_lat))
                if len(lat_hist) > 10:
                    lat_hist.pop(0)
                self.prev_lateral_dists[pair_key] = lat_hist

                if len(lat_hist) >= 4:
                    # Check 3 consecutive frame-to-frame lateral transitions
                    rates = [abs(lat_hist[i] - lat_hist[i - 1]) / dt_frame for i in range(-3, 0)]
                    sustained_encroachment = all(r >= 0.40 for r in rates)
                else:
                    sustained_encroachment = False

                if v_rel_kmh >= 12.0 and v_rel_mps > 1e-3:
                    ttc_long_a = d_long / v_rel_mps
                else:
                    ttc_long_a = float("inf")

                if sustained_encroachment and (ttc_long_a <= 1.50):
                    path_a = True

                # -------------------------------------------------------------
                # Candidate Path B (Critical Headway Compression with Deceleration):
                # - Longitudinal physical distance: d_long <= 5.0 m
                # - Longitudinal headway: TTC_headway = d_long / v_rel <= 0.80 s
                # - Active deceleration on the following vehicle: a_following <= -3.50 m/s^2
                # -------------------------------------------------------------
                path_b = False
                if d_long <= 5.0:
                    if v_rel_kmh >= 12.0 and v_rel_mps > 1e-3:
                        ttc_headway_b = d_long / v_rel_mps
                    elif v_rel_mps > 1e-3:
                        ttc_headway_b = d_long / v_rel_mps
                    else:
                        ttc_headway_b = float("inf")

                    if ttc_headway_b <= 0.80:
                        # Determine following vehicle based on longitudinal coordinate (traffic moving forward in Y)
                        # The following vehicle is at lower Y (or vehicle with higher speed closing the gap)
                        metric_a1_est = self._estimate_metric_acceleration(b1, k1.acceleration_mag)
                        metric_a2_est = self._estimate_metric_acceleration(b2, k2.acceleration_mag)
                        if p1_curr[1] < p2_curr[1]:
                            a_following = metric_a1_est
                        elif p2_curr[1] < p1_curr[1]:
                            a_following = metric_a2_est
                        else:
                            a_following = min(metric_a1_est, metric_a2_est)

                        if a_following <= -3.50:
                            path_b = True

                # -------------------------------------------------------------
                # Candidate Path C v3 (Multi-Frame Trajectory Divergence):
                # 1. Base Corridor Constraints: dist <= 8.0 m, d_lat < 1.80 m, ground history >= 6 frames
                # 2. Net Trajectory Divergence: |v_t - v_{t-4}| >= 15.0 deg
                # 3. Physical Lateral Kinematics: a_lat = v_t * (delta_theta_5f / delta_t_5f) <= 10.0 m/s^2
                # -------------------------------------------------------------
                path_c = False
                planar_dist = float(dist_curr)
                if planar_dist <= 8.0 and d_lat < 1.80 and len(hist1) >= 5 and len(hist2) >= 5:
                    def _evaluate_path_c_track(tr_obj: Track, k_obj: TrackKinematics, hist_obj: List[Tuple[float, float]]) -> Tuple[bool, float, float]:
                        # Must have ground history >= 6 frames (current position + 5 prior positions in history)
                        if len(hist_obj) < 5 or len(tr_obj.timestamps) < 5:
                            return False, 0.0, 0.0

                        # v_t: endpoint velocity over recent step (dt_curr)
                        p_t = k_obj.ground_pos
                        p_t1 = hist_obj[-1]
                        dt_curr = float((tr_obj.timestamps[-1] - tr_obj.timestamps[-2]) * 1e-9)
                        if dt_curr <= 1e-4:
                            dt_curr = 1.0 / 30.0

                        # v_{t-4}: endpoint velocity 4 steps prior (transition between t-5 and t-4)
                        p_t4 = hist_obj[-4]
                        p_t5 = hist_obj[-5]
                        dt_t4 = float((tr_obj.timestamps[-4] - tr_obj.timestamps[-5]) * 1e-9) if len(tr_obj.timestamps) >= 6 else (1.0 / 30.0)
                        if dt_t4 <= 1e-4:
                            dt_t4 = 1.0 / 30.0

                        v_curr_vec = np.array([(p_t[0] - p_t1[0]) / dt_curr, (p_t[1] - p_t1[1]) / dt_curr], dtype=np.float64)
                        v_t4_vec = np.array([(p_t4[0] - p_t5[0]) / dt_t4, (p_t4[1] - p_t5[1]) / dt_t4], dtype=np.float64)

                        norm_curr = float(np.linalg.norm(v_curr_vec))
                        norm_t4 = float(np.linalg.norm(v_t4_vec))

                        if norm_curr < 0.5 or norm_t4 < 0.5:
                            return False, 0.0, 0.0

                        dot_val = float(np.dot(v_curr_vec, v_t4_vec))
                        denom_val = norm_curr * norm_t4
                        cos_val = max(-1.0, min(1.0, dot_val / max(1e-6, denom_val)))
                        delta_theta_div = float(np.degrees(np.arccos(cos_val)))

                        # Time delta over 4 frames (t-4 to t)
                        dt_5f = float((tr_obj.timestamps[-1] - tr_obj.timestamps[-5]) * 1e-9)
                        if dt_5f <= 1e-4:
                            dt_5f = 4.0 / 30.0

                        v_speed_t = float(k_obj.speed_mps)
                        yaw_rate_rad = math.radians(delta_theta_div) / dt_5f
                        a_lat_val = v_speed_t * yaw_rate_rad

                        # Condition 2: Net angular change >= 15.0 deg
                        # Condition 3: Physical lateral acceleration <= 10.0 m/s^2 (~1.0 g)
                        passes = bool((delta_theta_div >= 15.0) and (a_lat_val <= 10.0))
                        return passes, delta_theta_div, a_lat_val

                    pass1, dtheta1, alat1 = _evaluate_path_c_track(tr1, k1, hist1)
                    pass2, dtheta2, alat2 = _evaluate_path_c_track(tr2, k2, hist2)

                    if pass1 or pass2:
                        path_c = True

                if path_a:
                    is_domain_2 = True
                    domain_2_path = "PATH_A"
                elif path_b:
                    is_domain_2 = True
                    domain_2_path = "PATH_B"
                elif path_c:
                    is_domain_2 = True
                    domain_2_path = "PATH_C"

            if is_domain_1:
                conflict_domain = "DOMAIN_1"
            elif is_domain_2:
                conflict_domain = f"DOMAIN_2_{domain_2_path}"
            else:
                self.log_suppressed(frame_idx, [t1_id, t2_id], "SUPPRESSED", d_lat)
                continue

            # Both Domain 1 and Domain 2 satisfy spatial and trajectory conflict prerequisites
            trajectory_conflict = True
            delta_d_lat = delta_d_lat_5f
            delta_theta_deg = delta_theta_5f

            # 5. Kinematic Response: Sudden Deceleration
            raw_a1 = k1.acceleration_mag
            raw_a2 = k2.acceleration_mag
            metric_a1 = self._estimate_metric_acceleration(b1, raw_a1)
            metric_a2 = self._estimate_metric_acceleration(b2, raw_a2)

            px_acc1 = getattr(k1, "pixel_acceleration", 0.0)
            px_acc2 = getattr(k2, "pixel_acceleration", 0.0)
            if px_acc1 == 0.0 and px_acc2 == 0.0:
                from road_surveillance.geometry.spatial_metrics import compute_pixel_kinematics
                _, px_acc1 = compute_pixel_kinematics(tr1, pts_ns)
                _, px_acc2 = compute_pixel_kinematics(tr2, pts_ns)

            norm_px_acc1 = (px_acc1 / max(15.0, max(b1[2], b1[3]))) * 4.5
            norm_px_acc2 = (px_acc2 / max(15.0, max(b2[2], b2[3]))) * 4.5

            valid_decel_1 = (-self.max_physical_decel <= metric_a1 < self.decel_threshold)
            valid_decel_2 = (-self.max_physical_decel <= metric_a2 < self.decel_threshold)
            valid_px_decel = (-self.max_physical_decel <= norm_px_acc1 < self.decel_threshold) or (
                -self.max_physical_decel <= norm_px_acc2 < self.decel_threshold
            )
            decel_signal = bool(valid_decel_1 or valid_decel_2 or valid_px_decel)

            h1_norm = math.hypot(*k1.heading_vector)
            h2_norm = math.hypot(*k2.heading_vector)
            cos_heading = float(np.dot(k1.heading_vector, k2.heading_vector)) if (h1_norm > 1e-4 and h2_norm > 1e-4) else 1.0
            is_opposing_head_on = (cos_heading < -0.60)
            is_one_wrong_way = getattr(tr1, "is_wrong_way", False) or getattr(tr2, "is_wrong_way", False)
            is_ww_suspect = False
            if self.wrong_way_handler is not None:
                mod = getattr(self.wrong_way_handler, "module", self.wrong_way_handler)
                vc = getattr(mod, "violation_counts", {})
                is_ww_suspect = (vc.get(t1_id, 0) > 0 or vc.get(t2_id, 0) > 0)

            if is_one_wrong_way or (is_opposing_head_on and is_ww_suspect):
                if self.wrong_way_handler is not None and hasattr(self.wrong_way_handler, "attach_near_miss"):
                    ww_id = t1_id if (getattr(tr1, "is_wrong_way", False) or (self.wrong_way_handler and getattr(getattr(self.wrong_way_handler, "module", None), "violation_counts", {}).get(t1_id, 0) > 0)) else t2_id
                    other_id = t2_id if ww_id == t1_id else t1_id
                    self.wrong_way_handler.attach_near_miss(
                        ww_id, other_id, frame_idx,
                        {"dca_m": conv.dca_m, "tca_sec": conv.tca_sec, "reason": "wrong_way_pass_near_miss"}
                    )
                self.log_suppressed(frame_idx, [t1_id, t2_id], "wrong_way_pass_near_miss", conv.dca_m)
                continue
            elif is_opposing_head_on:
                # Normal opposing traffic passing on two-way or divided carriageway without wrong-way violation
                self.log_suppressed(frame_idx, [t1_id, t2_id], "opposing_traffic_normal_pass", conv.dca_m)
                continue

            # 6. Meaningful Relative Speed
            relative_speed_kmh = float(np.hypot(
                k1.velocity_vector[0] - k2.velocity_vector[0],
                k1.velocity_vector[1] - k2.velocity_vector[1]
            ) * 3.6)
            is_closing = bool(conv.is_converging or tca_5f < 3.0 or delta_d_lat_5f < 0)
            closing_motion = bool(relative_speed_kmh >= self.min_closing_spd and is_closing)

            is_conflict = bool(trajectory_conflict and decel_signal and closing_motion)

            if not is_conflict:
                continue

            # Centroid and interaction bounding box
            centroid = (
                (b1[0] + b1[2] / 2.0 + b2[0] + b2[2] / 2.0) / 2.0,
                (b1[1] + b1[3] / 2.0 + b2[1] + b2[3] / 2.0) / 2.0,
            )
            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))

            # Temporal persistence verification
            matched = False
            triggered = False
            for cluster in self.spatial_clusters:
                if (frame_idx - cluster["last_frame"]) <= 3:
                    c_dist = np.hypot(centroid[0] - cluster["centroid"][0], centroid[1] - cluster["centroid"][1])
                    if c_dist <= 30.0:
                        cluster["persistence"] += 1
                        cluster["last_frame"] = frame_idx
                        cluster["centroid"] = centroid
                        cluster["box"] = interaction_box
                        matched = True
                        if cluster["persistence"] >= self.min_persistence:
                            triggered = True
                        break

            if not matched:
                self.spatial_clusters.append({
                    "centroid": centroid,
                    "box": interaction_box,
                    "last_frame": frame_idx,
                    "persistence": 1,
                })
                if self.min_persistence <= 1:
                    triggered = True

            if triggered:
                conf = 0.85
                candidates.append(
                    EventCandidate(
                        event_type="distant_trajectory_conflict",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[t1_id, t2_id],
                        primary_track_id=t1_id,
                        confidence=conf,
                        signals={
                            "trajectory_conflict": True,
                            "tca_sec": float(tca_5f),
                            "dca_m": float(dca_5f),
                            "tca_instantaneous": float(conv.tca_sec),
                            "dca_instantaneous": float(conv.dca_m),
                            "distance_m": float(dist),
                            "norm_distance": float(norm_dist),
                            "relative_speed_kmh": float(relative_speed_kmh),
                            "metric_accel_1": float(metric_a1),
                            "metric_accel_2": float(metric_a2),
                            "delta_theta_deg": float(delta_theta_deg),
                            "delta_d_lat": float(delta_d_lat),
                            "d_lat": float(d_lat),
                            "conflict_domain": str(conflict_domain),
                        },
                        ground_coordinates=centroid,
                        lane_id=k1.lane_id or k2.lane_id,
                        description=(
                            f"Distant trajectory conflict between tracks {t1_id} ({tr1.class_label}) and "
                            f"{t2_id} ({tr2.class_label}): TCA={tca_5f:.2f}s, DCA={dca_5f:.1f}m, "
                            f"rel_spd={relative_speed_kmh:.1f}km/h"
                        ),
                    )
                )

        # Cleanup disappeared pairs
        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
        for pk in list(self.prev_lateral_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_lateral_distances[pk]
        for pk in list(self.prev_lateral_dists.keys()):
            if pk not in current_pair_keys:
                del self.prev_lateral_dists[pk]
        for tid in list(self.prev_headings.keys()):
            if tid not in tracks:
                del self.prev_headings[tid]

        # Prune expired clusters
        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates
