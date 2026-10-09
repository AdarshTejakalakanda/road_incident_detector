"""Collision Incident Candidate Detector."""

import math
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence, calculate_directional_cosine
from ...calibration.lane_manager import LaneManager


class CollisionCandidateDetector(CandidateDetector):
    """Detects vehicle collision candidates using multi-signal kinematic convergence."""

    VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck", "van", "train", "vehicle", "candidate_vehicle_recovered"}

    def __init__(
        self,
        distance_decay_threshold_m: float = 1.5,
        tca_threshold_sec: float = 2.0,
        deceleration_threshold_ms2: float = -3.5,
        post_proximity_distance_m: float = 4.5,
        iou_overlap_threshold: float = 0.05,
        is_camera_calibrated: bool = False,
        min_persistence_frames: int = 1,
        min_track_age: int = 8,
    ):
        self.distance_decay_threshold_m = distance_decay_threshold_m
        self.tca_threshold_sec = tca_threshold_sec
        self.deceleration_threshold_ms2 = deceleration_threshold_ms2
        self.post_proximity_distance_m = post_proximity_distance_m
        self.iou_overlap_threshold = iou_overlap_threshold
        self.is_camera_calibrated = is_camera_calibrated
        self.min_persistence_frames = min_persistence_frames
        self.min_track_age = min_track_age
        self.max_collision_distance_m = 2.0
        self.wrong_way_handler = None
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_ious: Dict[Tuple[int, int], float] = {}
        self.prev_speeds: Dict[int, float] = {}
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.closure_persist_counts: Dict[Tuple[int, int], int] = {}
        self.last_closure_frame: Dict[Tuple[int, int], int] = {}
        self.track_speed_history: Dict[int, List[Tuple[int, int, float]]] = {}
        self.track_heading_history: Dict[int, List[Tuple[int, int, float]]] = {}
        self.dwell_counts: Dict[Tuple[int, int], int] = {}
        self.suppressed_events: List[Dict[str, Any]] = []

    def log_suppressed(self, frame_idx: int, tracks: List[int], reason: str, dist_or_dca: float) -> None:
        val = round(float(dist_or_dca), 2)
        self.suppressed_events.append({
            "frame": frame_idx,
            "tracks": list(tracks),
            "track_ids": list(tracks),
            "module": "collision_gate",
            "reason": reason,
            "distance_or_dca": val,
            "metric_val": val,
        })

    def reset(self) -> None:
        """Reset internal state between videos."""
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.prev_speeds.clear()
        self.persistence_counts.clear()
        self.closure_persist_counts.clear()
        self.last_closure_frame.clear()
        self.track_speed_history.clear()
        self.track_heading_history.clear()
        self.dwell_counts.clear()
        self.suppressed_events.clear()

    @staticmethod
    def _compute_bbox_iou_x(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Compute 1D horizontal IoU between two [x, y, w, h] boxes."""
        ax1, aw = boxA[0], boxA[2]
        bx1, bw = boxB[0], boxB[2]
        ax2 = ax1 + aw
        bx2 = bx1 + bw
        inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        union_x = max(ax2, bx2) - min(ax1, bx1)
        if union_x <= 1e-6:
            return 0.0
        return float(inter_x / union_x)

    @staticmethod
    def _compute_bbox_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """Compute IoU between two [x, y, w, h] boxes."""
        ax1, ay1, aw, ah = boxA
        bx1, by1, bw, bh = boxB
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        areaA = aw * ah
        areaB = bw * bh
        union = areaA + areaB - intersection
        if union <= 1e-6:
            return 0.0
        return float(intersection / union)

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
        current_pair_keys = set()

        # Update speed and heading histories for active tracks (Task R.1)
        for tid, k in kinematics.items():
            if tid not in self.track_speed_history:
                self.track_speed_history[tid] = []
            if tid not in self.track_heading_history:
                self.track_heading_history[tid] = []
            self.track_speed_history[tid].append((frame_idx, pts_ns, k.speed_kmh))
            self.track_heading_history[tid].append((frame_idx, pts_ns, k.heading_deg))
            if len(self.track_speed_history[tid]) > 60:
                self.track_speed_history[tid].pop(0)
            if len(self.track_heading_history[tid]) > 60:
                self.track_heading_history[tid].pop(0)

        for conv in convergences:
            t1_id = conv.track_id_1
            t2_id = conv.track_id_2
            k1 = kinematics.get(t1_id)
            k2 = kinematics.get(t2_id)
            tr1 = tracks.get(t1_id)
            tr2 = tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            # Strict vehicle class enforcement
            if tr1.class_label.lower() not in self.VEHICLE_LABELS or tr2.class_label.lower() not in self.VEHICLE_LABELS:
                continue

            # Track Maturity Gate: min(track_age) >= 8 frames
            if tr1.track_age < self.min_track_age or tr2.track_age < self.min_track_age:
                continue

            # Re-acquisition Settlement Gate: tracks recovering from coasting/lost must have >= 3 matched frames
            if (not getattr(tr1, "is_reacquisition_settled", True)) or (not getattr(tr2, "is_reacquisition_settled", True)):
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            # Safety Invariant: Real observation proximity
            # Both vehicles must have had at least one real detector observation within <= 3 frames
            if not (tr1.has_real_observation_near(-1, window=3) and tr2.has_real_observation_near(-1, window=3)):
                self.persistence_counts[pair_key] = 0
                continue

            # Check distance decrease
            dist = conv.distance_m
            prev_dist = self.prev_distances.get(pair_key, dist)
            distance_delta = dist - prev_dist
            self.prev_distances[pair_key] = dist

            # Signal 1: Decreasing distance
            signal_decreasing_dist = bool(distance_delta < -self.distance_decay_threshold_m)

            # Persistent closure tracking across consecutive frames
            is_closing = bool(distance_delta <= -0.05 or (signal_decreasing_dist and distance_delta < 0))
            if is_closing:
                self.closure_persist_counts[pair_key] = min(10, self.closure_persist_counts.get(pair_key, 0) + 1)
                self.last_closure_frame[pair_key] = frame_idx
            else:
                self.closure_persist_counts[pair_key] = max(0, self.closure_persist_counts.get(pair_key, 0) - 1)

            # Signal 4: Spatial proximity or 2D bounding box overlap
            iou = self._compute_bbox_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            # Proximity dwell tracking (bounded d <= 3.5m)
            if dist <= 3.5:
                self.dwell_counts[pair_key] = self.dwell_counts.get(pair_key, 0) + 1
            else:
                self.dwell_counts[pair_key] = 0
            dwell_frames = self.dwell_counts[pair_key]

            # Calculate heading alignment and delta
            cos_heading = calculate_directional_cosine(k1.heading_vector, k2.heading_vector)
            cos_clamped = max(-1.0, min(1.0, float(cos_heading)))
            heading_delta_deg = float(np.degrees(np.arccos(cos_clamped)))
            cos_align = cos_heading

            # Trajectory classification:
            # Genuine crossing/orthogonal trajectories: heading angle diverges from parallel (35 deg <= heading_delta_deg <= 145 deg)
            is_crossing_trajectory = (35.0 <= heading_delta_deg <= 145.0)
            heading_aligned = (cos_heading > 0.70)
            is_tbone_candidate = is_crossing_trajectory or (cos_align <= 0.70)
            is_opposing_head_on = (cos_heading < -0.70) or (not is_crossing_trajectory and cos_heading < -0.30)

            # Pixel geometry
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            c1_x = b1[0] + b1[2] / 2.0
            c1_y = b1[1] + b1[3] / 2.0
            c2_x = b2[0] + b2[2] / 2.0
            c2_y = b2[1] + b2[3] / 2.0
            px_dist = float(np.hypot(c1_x - c2_x, c1_y - c2_y))
            # Mandatory Safeguard 2: normalize Euclidean centroid distance by average bounding-box height of the pair
            avg_h = 0.5 * (b1[3] + b2[3])
            char_dim = max(15.0, avg_h)
            norm_pixel_dist = px_dist / char_dim

            mean_w = (b1[2] + b2[2]) / 2.0
            norm_lat_offset = abs(c1_x - c2_x) / max(1.0, mean_w)
            d_lateral = abs(k1.ground_pos[0] - k2.ground_pos[0])
            iou_x = self._compute_bbox_iou_x(b1, b2)

            # Task T.2: Persistent Parallel Rear-End Collision Candidate Reformulation
            is_parallel_heading = bool(heading_delta_deg <= 15.0 or abs(heading_delta_deg - 180.0) <= 15.0)
            shared_lateral_corridor = bool(d_lateral <= 1.8 or iou_x >= 0.50 or norm_lat_offset <= 1.5)

            had_persistent_closure = bool(
                self.closure_persist_counts.get(pair_key, 0) >= 3
                or (self.closure_persist_counts.get(pair_key, 0) >= 2 and (frame_idx - self.last_closure_frame.get(pair_key, -999)) <= 4)
            )

            # Proposal A: Terminal Physical-Contact Condition
            has_bbox_contact = bool(iou >= 0.15 or (iou >= 0.10 and iou_x >= 0.50))
            has_calibrated_ground_contact = bool(self.is_camera_calibrated and dist <= 2.0)
            has_uncalibrated_pixel_proximity = bool(not self.is_camera_calibrated and (norm_pixel_dist <= 0.50 or (norm_pixel_dist <= 0.85 and iou > 0.0)))

            has_terminal_physical_contact = bool(
                has_bbox_contact or has_calibrated_ground_contact or has_uncalibrated_pixel_proximity
            )

            has_contact_distance_sanity = bool(
                (dist <= 3.5) if self.is_camera_calibrated else (norm_pixel_dist <= 1.0 or dist <= 4.0)
            )

            # Mandatory Safeguard 1: Signed longitudinal deceleration, never unsigned magnitude
            a1_long = k1.acceleration_mag
            a2_long = k2.acceleration_mag
            has_compressive_deceleration = bool(
                a1_long <= -6.0
                or a2_long <= -6.0
                or a1_long <= -12.0
                or a2_long <= -12.0
            )

            is_kinematic_closure_corridor = bool(
                is_parallel_heading
                and shared_lateral_corridor
                and had_persistent_closure
                and has_terminal_physical_contact
                and has_contact_distance_sanity
                and has_compressive_deceleration
            )

            # Task R.1: Composite Kinetic Evidence for Crossing Candidates (2.0m < d <= 3.2m)
            # 1. Velocity Collapse Signal (S_v_drop)
            def compute_v_drop(tid, curr_spd):
                hist = self.track_speed_history.get(tid, [])
                if not hist:
                    return 0.0
                t_now = pts_ns
                window_speeds = [h[2] for h in hist if (t_now - h[1]) <= 1.2e9 or (frame_idx - h[0]) <= 36]
                if not window_speeds:
                    window_speeds = [h[2] for h in hist[-36:]]
                v_pre = max(window_speeds)
                if v_pre <= 3.0:
                    return 0.0
                drop_ratio = max(0.0, (v_pre - curr_spd) / v_pre)
                return min(1.0, drop_ratio / 0.40)

            s_v_drop = max(compute_v_drop(t1_id, k1.speed_kmh), compute_v_drop(t2_id, k2.speed_kmh))

            # 2. Trajectory Heading Deflection (S_deflect)
            def compute_heading_deflect(tid, curr_hdg):
                hist = self.track_heading_history.get(tid, [])
                if len(hist) < 3:
                    return 0.0
                t_now = pts_ns
                window_headings = [h[2] for h in hist if (t_now - h[1]) <= 1.2e9 or (frame_idx - h[0]) <= 36]
                if not window_headings:
                    window_headings = [h[2] for h in hist[-36:]]
                pre_hdg = window_headings[0]
                dh = abs(curr_hdg - pre_hdg)
                if dh > 180.0:
                    dh = 360.0 - dh
                return min(1.0, dh / 25.0)

            s_deflect = max(compute_heading_deflect(t1_id, k1.heading_deg), compute_heading_deflect(t2_id, k2.heading_deg))

            # 3. Proximity Persistence Dwell (S_dwell)
            s_dwell = 1.0 if dwell_frames >= 5 else 0.0

            # 4. Kinetic Rest Support (S_rest)
            min_post_spd = min(k1.speed_kmh, k2.speed_kmh)
            s_rest = 1.0 if min_post_spd <= 5.0 else 0.0

            # Composite Admission Gate:
            s_dissip = 0.35 * s_v_drop + 0.35 * s_deflect + 0.20 * s_dwell + 0.10 * s_rest

            # Q.2 & Q.3 Trajectory-aware collision distance:
            is_one_wrong_way = getattr(tr1, "is_wrong_way", False) or getattr(tr2, "is_wrong_way", False)
            if is_crossing_trajectory and not is_one_wrong_way:
                selected_max_dist_m = 3.2
            else:
                selected_max_dist_m = 2.0

            # Max collision distance sanity check:
            if not is_kinematic_closure_corridor:
                if dist > selected_max_dist_m and iou < 0.10:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue

            # Task R.1 Composite Admission Gate for Crossing Candidates:
            # If d <= 2.0 m: Admit under standard collision geometry.
            # If 2.0 m < d <= 3.2 m: Require S_dissip >= 0.55 for crossing candidates.
            if 2.0 < dist <= 3.2 and is_crossing_trajectory:
                if s_dissip < 0.55 and iou < 0.25:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "kinetic_dissipation_insufficient", dist)
                    continue

            # Multi-vehicle collision requires standard vehicle tracks or mature interaction-rescued tracks
            is_immature_cand = False
            for tr_chk in (tr1, tr2):
                if getattr(tr_chk, "is_tentative_rescued", False) and tr_chk.track_age < 3:
                    is_immature_cand = True
                    break
                if (getattr(tr_chk, "fused_class", "") == "candidate_vehicle_recovered" or tr_chk.class_label == "candidate_vehicle_recovered") and tr_chk.track_age < 2:
                    is_immature_cand = True
                    break
            if is_immature_cand:
                continue

            # Wrong-way vehicle interaction gate:
            # If either track is wrong-way or head-on passing contrary to flow, do not trigger collision purely from passing proximity.
            # Multi-vehicle collision gate:
            # Suppress collision proposals between opposing traffic and wrong-way vehicles
            # when there is no physical impact (physical_overlap and (shape_deformation or velocity_collapse)).
            # Q.1 Decouple flow coherence from cross-traffic collision:
            # If crossing trajectories: collision admission MUST NOT depend on background traffic-flow coherence.
            # ignore_flow_coherence_for_collision = True.
            is_ww_suspect = False
            if not is_crossing_trajectory and self.wrong_way_handler is not None:
                mod = getattr(self.wrong_way_handler, "module", self.wrong_way_handler)
                vc = getattr(mod, "violation_counts", {})
                is_ww_suspect = (vc.get(t1_id, 0) > 0 or vc.get(t2_id, 0) > 0)
                if not is_ww_suspect and hasattr(mod, "flow_field"):
                    for tr_c, k_c in [(tr1, k1), (tr2, k2)]:
                        gc = tr_c.calculate_ground_contact(tr_c.current_bbox_xywh)
                        fl_m = mod.flow_field.get_flow_model_for_position(gc[0] / 1280.0, gc[1] / 720.0)
                        if fl_m and fl_m.is_persisted:
                            c_flow = calculate_directional_cosine(k_c.heading_vector, fl_m.direction_vector)
                            if c_flow <= -0.40:
                                is_ww_suspect = True
                                break

            if not is_crossing_trajectory and (is_one_wrong_way or is_ww_suspect) and is_opposing_head_on:
                # Physical overlap requires true physical proximity on ground (d <= 1.0m, or d <= 2.0m with 2D IoU >= 0.10)
                physical_overlap = (dist <= 1.0 or (dist <= 2.0 and iou >= 0.10))

                def compute_ar_delta(tr_obj):
                    if len(tr_obj.bbox_history) >= 2:
                        w1, h1 = tr_obj.bbox_history[-1][2], tr_obj.bbox_history[-1][3]
                        w0, h0 = tr_obj.bbox_history[-2][2], tr_obj.bbox_history[-2][3]
                        ar1 = w1 / max(1.0, h1)
                        ar0 = w0 / max(1.0, h0)
                        return abs(ar1 - ar0) / max(0.1, ar0)
                    return 0.0

                shape_deformation = (compute_ar_delta(tr1) >= 0.25 or compute_ar_delta(tr2) >= 0.25)

                def check_velocity_collapse(tr_id, tr_obj, k_obj):
                    # 1. speed drop >= 15.0 km/h within 3 frames ending at near-zero (< 3.0 km/h)
                    curr_spd = k_obj.speed_kmh
                    if curr_spd < 3.0:
                        prev_spd = self.prev_speeds.get(tr_id, curr_spd)
                        if (prev_spd - curr_spd) >= 15.0:
                            return True
                        for h in getattr(tr_obj, "ground_history", [])[-4:]:
                            h_spd = getattr(h, "speed_kmh", None)
                            if h_spd is not None and (h_spd - curr_spd) >= 15.0:
                                return True
                    # 2. sudden deceleration > 15.0 m/s^2 with post-event arrest (tau_rest >= 0.25 s)
                    if abs(k_obj.acceleration_mag) >= 15.0 and getattr(k_obj, "stationary_duration_sec", 0.0) >= 0.25:
                        return True
                    return False

                velocity_collapse = (check_velocity_collapse(t1_id, tr1, k1) or check_velocity_collapse(t2_id, tr2, k2))

                has_impact = physical_overlap and (shape_deformation or velocity_collapse)
                if not has_impact:
                    ww_id = t1_id if (getattr(tr1, "is_wrong_way", False) or (self.wrong_way_handler and getattr(getattr(self.wrong_way_handler, "module", None), "violation_counts", {}).get(t1_id, 0) > 0)) else t2_id
                    other_id = t2_id if ww_id == t1_id else t1_id
                    if self.wrong_way_handler is not None and hasattr(self.wrong_way_handler, "attach_near_miss"):
                        self.wrong_way_handler.attach_near_miss(
                            ww_id, other_id, frame_idx,
                            {"distance_m": dist, "dca_m": conv.dca_m, "iou": iou, "reason": "safe_oncoming_pass"}
                        )
                    self.persistence_counts[pair_key] = 0
                    reason_str = "wrong_way_pass_near_miss" if (is_one_wrong_way or is_ww_suspect) else "collision_distance_exceeded"
                    self.log_suppressed(frame_idx, [t1_id, t2_id], reason_str, dist)
                    continue

            # Pixel geometry
            c1_x = tr1.current_bbox_xywh[0] + tr1.current_bbox_xywh[2] / 2.0
            c1_y = tr1.current_bbox_xywh[1] + tr1.current_bbox_xywh[3] / 2.0
            c2_x = tr2.current_bbox_xywh[0] + tr2.current_bbox_xywh[2] / 2.0
            px_dist = float(np.hypot(c1_x - c2_x, c1_y - c2_y))
            avg_h = 0.5 * (tr1.current_bbox_xywh[3] + tr2.current_bbox_xywh[3])
            char_dim = max(15.0, avg_h)
            norm_pixel_dist = px_dist / char_dim

            # Safeguard 1: Infant duplicate tracklet guard
            # If either tracklet was born this frame (age <= 1) and overlaps with an existing track,
            # it is an internal duplicate detection split, not an independent vehicle collision.
            is_infant_duplicate = ((tr1.track_age >= 2 and tr2.track_age == 1) or (tr2.track_age >= 2 and tr1.track_age == 1)) and (iou >= 0.15 or norm_pixel_dist <= 0.40)
            if is_infant_duplicate:
                self.persistence_counts[pair_key] = 0
                continue

            # Heading alignment
            cos_align = calculate_directional_cosine(k1.heading_vector, k2.heading_vector)
            heading_aligned = cos_align > 0.70
            is_tbone_candidate = cos_align <= 0.70

            # P.1 Explicit angular convergence using normalized velocity vectors
            v1 = np.array(k1.velocity_vector, dtype=np.float64)
            v2 = np.array(k2.velocity_vector, dtype=np.float64)
            spd1_mps = float(np.linalg.norm(v1))
            spd2_mps = float(np.linalg.norm(v2))

            p1 = np.array(k1.ground_pos, dtype=np.float64)
            p2 = np.array(k2.ground_pos, dtype=np.float64)
            r12 = p2 - p1
            dist_pos = float(np.linalg.norm(r12))

            if dist_pos > 1e-3:
                u_r12 = r12 / dist_pos
            else:
                u_r12 = np.array([0.0, 1.0], dtype=np.float64)

            u_v1 = v1 / spd1_mps if spd1_mps > 0.3 else np.array([0.0, 0.0], dtype=np.float64)
            u_v2 = v2 / spd2_mps if spd2_mps > 0.3 else np.array([0.0, 0.0], dtype=np.float64)

            # Projections along the line of sight connecting them:
            # proj1 > 0 means track 1 has velocity component heading towards track 2
            # proj2 > 0 means track 2 has velocity component heading towards track 1
            proj1 = float(np.dot(u_v1, u_r12)) if spd1_mps > 0.3 else 0.0
            proj2 = float(np.dot(u_v2, -u_r12)) if spd2_mps > 0.3 else 0.0

            # Closing velocity along the line of sight (m/s, positive = closing)
            closing_velocity_mps = float(np.dot(v1 - v2, u_r12))
            has_angular_approach = (proj1 > 0.10 or proj2 > 0.10) and (closing_velocity_mps > 0.3 or distance_delta < -self.distance_decay_threshold_m)

            # Safeguard 2: Duplicate tracks on same vehicle
            # Two tracks on the same physical vehicle travel in the same direction, have high overlap,
            # and their distance is not closing from far away.
            is_duplicate = (iou >= 0.25 or (dist < 1.2 and norm_pixel_dist <= 0.35)) and heading_aligned and not signal_decreasing_dist
            if is_duplicate:
                self.persistence_counts[pair_key] = 0
                continue

            # Signal 3: Deceleration anomaly
            decel_1 = k1.acceleration_mag < self.deceleration_threshold_ms2
            decel_2 = k2.acceleration_mag < self.deceleration_threshold_ms2
            signal_sudden_decel = bool(decel_1 or decel_2)
            severe_decel = bool(k1.acceleration_mag < -6.0 or k2.acceleration_mag < -6.0)

            # Signal 2: Trajectory convergence (TCA and DCA)
            has_motion = (k1.speed_kmh >= 2.5 or k2.speed_kmh >= 2.5)
            if self.is_camera_calibrated:
                signal_convergence = bool(has_motion and conv.is_converging and conv.tca_sec < self.tca_threshold_sec and conv.dca_m < self.post_proximity_distance_m)
                signal_proximity = bool(dist < self.post_proximity_distance_m or (iou > self.iou_overlap_threshold and dist < 12.0))
            else:
                signal_proximity = bool(iou >= self.iou_overlap_threshold or norm_pixel_dist <= 0.8)
                signal_convergence = bool(has_motion and conv.is_converging and conv.tca_sec < self.tca_threshold_sec)

            # Safeguard 3: Parallel same-direction overtaking guard
            # In same-direction traffic, passing or adjacent lane driving without physical contact (iou < 0.25)
            # or severe impact deceleration is normal highway flow, not a collision.
            if heading_aligned and not severe_decel and iou < 0.25:
                self.persistence_counts[pair_key] = 0
                continue

            # P.1 T-bone candidate filtering:
            # Require genuine approach/convergence rather than centroid proximity alone for T-bone candidates.
            if is_tbone_candidate and not is_kinematic_closure_corridor:
                has_tbone_approach = (
                    (conv.is_converging and conv.tca_sec < self.tca_threshold_sec and conv.dca_m < 3.5 and has_angular_approach)
                    or (closing_velocity_mps > 0.8 and max(proj1, proj2) > 0.20 and dist < 5.0)
                    or (iou >= 0.15)
                )
                if not has_tbone_approach:
                    self.persistence_counts[pair_key] = 0
                    continue

            # P.1 Post-event rest requirement where applicable:
            # If both vehicles maintain steady cruising speeds without deceleration or stop,
            # they are passing traffic, not a physical collision.
            both_cruising_without_arrest = (
                k1.speed_kmh >= 15.0 and k2.speed_kmh >= 15.0
                and k1.stationary_duration_sec < 0.5
                and k2.stationary_duration_sec < 0.5
                and not (signal_sudden_decel and (signal_decreasing_dist or signal_convergence))
            )
            if both_cruising_without_arrest and not is_kinematic_closure_corridor:
                self.persistence_counts[pair_key] = 0
                continue

            # Multi-vehicle collision ground distance gate (d_ground <= selected_max_dist_m or physical impact or kinematic closure corridor)
            if not is_kinematic_closure_corridor:
                if dist > selected_max_dist_m and not (iou >= 0.20 and severe_decel and (signal_decreasing_dist or signal_convergence)):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue

            # Multi-signal triggering rule
            # In aligned/parallel flow, require genuine physical approach rather than 2D perspective box overlap
            if heading_aligned:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach)
            else:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach or (iou >= 0.20))

            if is_kinematic_closure_corridor:
                signal_proximity = True
                has_physical_approach = True
                is_candidate_frame = True
            else:
                is_candidate_frame = signal_proximity and has_physical_approach and (signal_convergence or signal_sudden_decel or signal_decreasing_dist or (iou >= 0.15))

            if is_candidate_frame:
                self.persistence_counts[pair_key] = self.persistence_counts.get(pair_key, 0) + 1
            else:
                self.persistence_counts[pair_key] = 0

            # Trigger condition: require persistence across frames (defaults to 2 in production)
            is_collision_candidate = self.persistence_counts[pair_key] >= self.min_persistence_frames

            if is_collision_candidate:
                center_g = (
                    (k1.ground_pos[0] + k2.ground_pos[0]) / 2.0,
                    (k1.ground_pos[1] + k2.ground_pos[1]) / 2.0,
                )
                active_signals = {
                    "decreasing_distance": signal_decreasing_dist,
                    "distance_delta_m": distance_delta,
                    "trajectory_convergence": signal_convergence,
                    "tca_sec": conv.tca_sec,
                    "dca_m": conv.dca_m,
                    "sudden_deceleration": signal_sudden_decel,
                    "accel_1_ms2": k1.acceleration_mag,
                    "accel_2_ms2": k2.acceleration_mag,
                    "post_event_proximity": signal_proximity,
                    "ground_distance_m": dist,
                    "distance_m": dist,
                    "heading_delta_deg": heading_delta_deg,
                    "is_crossing_trajectory": is_crossing_trajectory,
                    "selected_max_dist_m": selected_max_dist_m,
                    "ignore_flow_coherence": is_crossing_trajectory,
                    "bbox_iou": iou,
                    "persistence_frames": self.persistence_counts[pair_key],
                    "angular_approach": has_angular_approach,
                    "closing_velocity_mps": closing_velocity_mps,
                    "proj1": proj1,
                    "proj2": proj2,
                    "is_tbone": is_tbone_candidate,
                    "cos_align": cos_align,
                    "post_event_rest": (k1.stationary_duration_sec >= 1.5 or k2.stationary_duration_sec >= 1.5 or k1.speed_kmh < 3.0 or k2.speed_kmh < 3.0 or severe_decel),
                    "S_dissip": round(s_dissip, 4),
                    "S_v_drop": round(s_v_drop, 4),
                    "S_deflect": round(s_deflect, 4),
                    "S_dwell": round(s_dwell, 4),
                    "S_rest": round(s_rest, 4),
                    "dist_m": dist,
                    "heading_delta": heading_delta_deg,
                    "norm_pixel_dist": round(norm_pixel_dist, 3),
                    "is_kinematic_closure_corridor": is_kinematic_closure_corridor,
                    "is_tentative_rescued": bool(getattr(tr1, "is_tentative_rescued", False) or getattr(tr2, "is_tentative_rescued", False)),
                    "pixel_centroid": ((c1_x + c2_x) / 2.0, (c1_y + c2_y) / 2.0),
                    "char_dim": char_dim,
                }
                active_signal_count = sum([signal_decreasing_dist, signal_convergence, signal_sudden_decel, signal_proximity])
                conf = min(0.98, 0.4 + 0.15 * active_signal_count)
                candidates.append(
                    EventCandidate(
                        event_type="collision_candidate",
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[t1_id, t2_id],
                        primary_track_id=t1_id,
                        confidence=conf,
                        signals=active_signals,
                        ground_coordinates=center_g,
                        lane_id=k1.lane_id or k2.lane_id,
                        description=(
                            f"Collision candidate between tracks {t1_id} ({tr1.class_label}) and "
                            f"{t2_id} ({tr2.class_label}): dist={dist:.1f}m, iou={iou:.2f}, "
                            f"active_signals={active_signal_count}/4 (persistence={self.persistence_counts[pair_key]}f)"
                        ),
                    )
                )

        # Update speed memory for active tracks
        for tid, k in kinematics.items():
            self.prev_speeds[tid] = k.speed_kmh

        # Cleanup disappeared pairs
        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
        for pk in list(self.prev_ious.keys()):
            if pk not in current_pair_keys:
                del self.prev_ious[pk]
        for pk in list(self.persistence_counts.keys()):
            if pk not in current_pair_keys:
                del self.persistence_counts[pk]
        for pk in list(self.dwell_counts.keys()):
            if pk not in current_pair_keys:
                del self.dwell_counts[pk]
        for tid in list(self.prev_speeds.keys()):
            if tid not in tracks:
                del self.prev_speeds[tid]
                self.track_speed_history.pop(tid, None)
                self.track_heading_history.pop(tid, None)

        return candidates

    def reset(self) -> None:
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.persistence_counts.clear()
