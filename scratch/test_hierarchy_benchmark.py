import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set

from road_surveillance.config import CollisionConfig
from road_surveillance.tracking import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from scripts.benchmark_iteration_b_detector_recovery import (
    compute_iou,
    cluster_episodes,
    evaluate_video_candidates_temporal,
    RestoredIncidentRegion,
)

# 1. Pixel kinematics helper
def get_pixel_kinematics(tr: Track, pts_ns: int) -> Tuple[float, float]:
    if pts_ns in tr.timestamps:
        idx = tr.timestamps.index(pts_ns)
        pos_hist = tr.position_history[:idx+1]
        time_hist = tr.timestamps[:idx+1]
    else:
        pos_hist = tr.position_history
        time_hist = tr.timestamps
    px_spd = 0.0
    px_acc = 0.0
    if len(pos_hist) >= 2 and len(time_hist) >= 2:
        dt = (time_hist[-1] - time_hist[-2]) * 1e-9
        if dt > 1e-5:
            dp = np.hypot(pos_hist[-1][0] - pos_hist[-2][0], pos_hist[-1][1] - pos_hist[-2][1])
            px_spd = dp / dt
            if len(pos_hist) >= 3 and len(time_hist) >= 3:
                dt_prev = (time_hist[-2] - time_hist[-3]) * 1e-9
                if dt_prev > 1e-5:
                    dp_prev = np.hypot(pos_hist[-2][0] - pos_hist[-3][0], pos_hist[-2][1] - pos_hist[-3][1])
                    px_spd_prev = dp_prev / dt_prev
                    px_acc = (px_spd - px_spd_prev) / dt
    return px_spd, px_acc


# 2. Refined Lifecycle Manager
class RefinedIncidentLifecycleManager:
    def __init__(self, is_camera_calibrated: bool = False):
        self.is_camera_calibrated = is_camera_calibrated
        self.incidents: List[RestoredIncidentRegion] = []
        self.next_id = 1

    def reset(self):
        self.incidents.clear()
        self.next_id = 1

    def evaluate_candidate_interaction(
        self,
        frame_idx: int,
        t1_id: int,
        t2_id: int,
        tr1: Track,
        tr2: Track,
        k1: TrackKinematics,
        k2: TrackKinematics,
        px_centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[str]]:
        for inc in self.incidents:
            if inc.state == "CLEARED":
                continue

            c_dist = np.hypot(px_centroid[0] - inc.centroid[0], px_centroid[1] - inc.centroid[1])
            box_iou = compute_iou(interaction_bbox, inc.bbox)

            spatial_radius = 8.0 if self.is_camera_calibrated else max(150.0, 2.0 * max(inc.bbox[2], inc.bbox[3]))
            center_inside = (inc.bbox[0] <= px_centroid[0] <= inc.bbox[0] + inc.bbox[2]) and (
                inc.bbox[1] <= px_centroid[1] <= inc.bbox[1] + inc.bbox[3]
            )
            is_in_zone = (c_dist <= spatial_radius) or (box_iou >= 0.05) or center_inside

            if is_in_zone:
                t1_in_inc = (t1_id in inc.involved_track_ids) or (t1_id in inc.resting_track_ids)
                t2_in_inc = (t2_id in inc.involved_track_ids) or (t2_id in inc.resting_track_ids)

                if self.is_camera_calibrated:
                    s1_low = k1.speed_kmh < 6.0
                    s2_low = k2.speed_kmh < 6.0
                else:
                    px_spd1, _ = get_pixel_kinematics(tr1, k1.pts_ns)
                    px_spd2, _ = get_pixel_kinematics(tr2, k2.pts_ns)
                    char1 = max(15.0, max(tr1.current_bbox_xywh[2], tr1.current_bbox_xywh[3]))
                    char2 = max(15.0, max(tr2.current_bbox_xywh[2], tr2.current_bbox_xywh[3]))
                    s1_low = (px_spd1 < 35.0) or ((px_spd1 / char1) < 0.4)
                    s2_low = (px_spd2 < 35.0) or ((px_spd2 / char2) < 0.4)

                # Damaged / Resting Wreckage Jitter
                if (t1_in_inc and t2_in_inc) or (s1_low and s2_low):
                    inc.resting_track_ids.add(t1_id)
                    inc.resting_track_ids.add(t2_id)
                    inc.state = "POST_IMPACT_RESTING"
                    inc.last_active_frame = frame_idx
                    inc.update_bounds(interaction_bbox)
                    return True, "WRECKAGE_RESTING"

                # Passing Vehicle Grazing Stationary Wreckage
                if (t1_in_inc and not t2_in_inc) or (t2_in_inc and not t1_in_inc):
                    inc.last_active_frame = frame_idx
                    return True, "PASSING_VEHICLE_RESTING_INTERACTION"

                # Post-impact spatial zone absorption within 800 frames
                if (frame_idx - inc.origin_frame) <= 800:
                    inc.last_active_frame = frame_idx
                    inc.update_bounds(interaction_bbox)
                    return True, "POST_IMPACT_SPATIAL_ZONE"

        return False, None

    def register_incident(
        self,
        frame_idx: int,
        t1_id: int,
        t2_id: int,
        px_centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ):
        bx, by, bw, bh = interaction_bbox
        mw = bw * 0.4
        mh = bh * 0.4
        expanded = (max(0.0, bx - mw), max(0.0, by - mh), bw + 2 * mw, bh + 2 * mh)

        inc = RestoredIncidentRegion(
            region_id=self.next_id,
            origin_frame=frame_idx,
            centroid=px_centroid,
            bbox=expanded,
            involved_track_ids={t1_id, t2_id},
        )
        self.incidents.append(inc)
        self.next_id += 1


# 3. Refined Pipeline
class RefinedProductionPipeline:
    def __init__(self, is_camera_calibrated: bool = False):
        self.is_camera_calibrated = is_camera_calibrated
        self.cfg = CollisionConfig(
            max_ground_distance_m=4.0,
            max_perspective_distance_m=12.0,
            min_iou_overlap=0.15,
            min_persistence_frames=2,
            temporal_cluster_gap_frames=25,
        )
        self.lifecycle_mgr = RefinedIncidentLifecycleManager(is_camera_calibrated=is_camera_calibrated)
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_ious: Dict[Tuple[int, int], float] = {}
        self.single_track_history: Dict[int, List[Dict[str, Any]]] = {}

    def reset(self):
        self.lifecycle_mgr.reset()
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.single_track_history.clear()

    def _is_in_lifecycle_incident(
        self,
        centroid: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        tids: List[int],
    ) -> bool:
        for inc in self.lifecycle_mgr.incidents:
            if inc.state == "CLEARED":
                continue
            c_dist = np.hypot(centroid[0] - inc.centroid[0], centroid[1] - inc.centroid[1])
            spatial_radius = 8.0 if self.is_camera_calibrated else max(150.0, 2.0 * max(inc.bbox[2], inc.bbox[3]))
            box_iou = compute_iou(bbox, inc.bbox)
            center_inside = (inc.bbox[0] <= centroid[0] <= inc.bbox[0] + inc.bbox[2]) and (
                inc.bbox[1] <= centroid[1] <= inc.bbox[1] + inc.bbox[3]
            )
            tid_overlap = any(t in inc.involved_track_ids or t in inc.resting_track_ids for t in tids)
            if c_dist <= spatial_radius or box_iou >= 0.05 or center_inside or tid_overlap:
                return True
        return False

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
    ) -> List[Dict[str, Any]]:
        frame_candidates: List[Dict[str, Any]] = []
        if frame_idx < 15:
            return frame_candidates

        active_collision_tids: Set[int] = set()
        VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

        # -------------------------------------------------------------
        # 1. Detector B: Pairwise Collision with Kinematic Hierarchy
        # -------------------------------------------------------------
        for conv in convergences:
            t1_id, t2_id = conv.track_id_1, conv.track_id_2
            k1, k2 = kinematics.get(t1_id), kinematics.get(t2_id)
            tr1, tr2 = tracks.get(t1_id), tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue
            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            iou = compute_iou(b1, b2)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            c1 = (b1[0] + b1[2]/2.0, b1[1] + b1[3]/2.0)
            c2 = (b2[0] + b2[2]/2.0, b2[1] + b2[3]/2.0)
            px_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_px_dist = px_dist / char_dim

            # Track maturity
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            # Ground coordinates validity check
            gx1, gy1 = k1.ground_pos
            gx2, gy2 = k2.ground_pos
            g_valid1 = (abs(gx1) <= 150.0 and -50.0 <= gy1 <= 250.0)
            g_valid2 = (abs(gx2) <= 150.0 and -50.0 <= gy2 <= 250.0)

            # Pixel kinematics
            px_spd1, px_acc1 = get_pixel_kinematics(tr1, pts_ns)
            px_spd2, px_acc2 = get_pixel_kinematics(tr2, pts_ns)
            norm_px_acc1 = (px_acc1 / max(15.0, max(b1[2], b1[3]))) * 4.5
            norm_px_acc2 = (px_acc2 / max(15.0, max(b2[2], b2[3]))) * 4.5

            # Spatial proximity check
            if self.is_camera_calibrated and g_valid1 and g_valid2:
                proximity_signal = (conv.distance_m <= self.cfg.max_ground_distance_m) or (
                    iou >= self.cfg.min_iou_overlap and conv.distance_m <= self.cfg.max_perspective_distance_m
                )
            else:
                # In uncalibrated mode, MUST satisfy image-space proximity
                # (rejects distant cross-highway pairs > 500 px / > 5.0 body-lengths apart)
                proximity_signal = (iou >= self.cfg.min_iou_overlap) or (norm_px_dist <= 2.5 and px_dist <= 450.0)

            if not proximity_signal:
                continue

            # Kinematic Deceleration signal
            raw_a1 = k1.acceleration_mag * (4.5 / max(15.0, max(b1[2], b1[3])))
            raw_a2 = k2.acceleration_mag * (4.5 / max(15.0, max(b2[2], b2[3])))
            decel_signal = (
                (-60.0 <= raw_a1 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= raw_a2 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= norm_px_acc1 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= norm_px_acc2 <= self.cfg.deceleration_threshold_ms2)
            )

            prev_d = self.prev_distances.get(pair_key, px_dist)
            is_closing = (px_dist - prev_d) < -1.0
            self.prev_distances[pair_key] = px_dist
            conv_signal = bool(is_closing or iou_increasing)

            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))
            px_centroid = (ux1 + (ux2 - ux1)/2.0, uy1 + (uy2 - uy1)/2.0)

            is_redundant, _ = self.lifecycle_mgr.evaluate_candidate_interaction(
                frame_idx, t1_id, t2_id, tr1, tr2, k1, k2, px_centroid, interaction_box
            )
            if is_redundant:
                continue

            if proximity_signal and conv_signal and decel_signal:
                self.lifecycle_mgr.register_incident(
                    frame_idx, t1_id, t2_id, px_centroid, interaction_box
                )
                frame_candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [t1_id, t2_id],
                    "dist": float(conv.distance_m),
                    "iou": float(iou),
                    "accel_1": float(k1.acceleration_mag),
                    "accel_2": float(k2.acceleration_mag),
                    "detector": "pairwise_collision",
                })
                active_collision_tids.update([t1_id, t2_id])

        # -------------------------------------------------------------
        # 2. Detector D: Distant Trajectory Conflict
        # -------------------------------------------------------------
        for conv in convergences:
            t1_id, t2_id = conv.track_id_1, conv.track_id_2
            if t1_id in active_collision_tids and t2_id in active_collision_tids:
                continue
            k1, k2 = kinematics.get(t1_id), kinematics.get(t2_id)
            tr1, tr2 = tracks.get(t1_id), tracks.get(t2_id)
            if not k1 or not k2 or not tr1 or not tr2:
                continue
            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            if b1[0] < 15.0 or b1[1] < 15.0 or b2[0] < 15.0 or b2[1] < 15.0:
                continue

            iou = compute_iou(b1, b2)
            c1 = (b1[0] + b1[2]/2.0, b1[1] + b1[3]/2.0)
            c2 = (b2[0] + b2[2]/2.0, b2[1] + b2[3]/2.0)
            px_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_px_dist = px_dist / char_dim

            # Spatial distance: 0.8 to 3.5 body lengths, within 450 px
            if not (0.8 <= norm_px_dist <= 3.5 and px_dist <= 450.0):
                continue

            # Trajectory conflict check
            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            prev_d = self.prev_distances.get(pair_key, px_dist)
            is_closing = (px_dist - prev_d) < -1.0
            if not is_closing:
                continue

            px_spd1, px_acc1 = get_pixel_kinematics(tr1, pts_ns)
            px_spd2, px_acc2 = get_pixel_kinematics(tr2, pts_ns)
            norm_px_acc1 = (px_acc1 / max(15.0, max(b1[2], b1[3]))) * 4.5
            norm_px_acc2 = (px_acc2 / max(15.0, max(b2[2], b2[3]))) * 4.5

            decel_signal = (norm_px_acc1 <= -3.0) or (norm_px_acc2 <= -3.0)
            if not decel_signal:
                continue

            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))
            px_centroid = (ux1 + (ux2 - ux1)/2.0, uy1 + (uy2 - uy1)/2.0)

            if self._is_in_lifecycle_incident(px_centroid, interaction_box, [t1_id, t2_id]):
                continue

            self.lifecycle_mgr.register_incident(frame_idx, t1_id, t2_id, px_centroid, interaction_box)
            frame_candidates.append({
                "frame_idx": frame_idx,
                "pts_ns": pts_ns,
                "track_ids": [t1_id, t2_id],
                "dist": float(conv.distance_m),
                "iou": float(iou),
                "accel_1": float(k1.acceleration_mag),
                "accel_2": float(k2.acceleration_mag),
                "detector": "distant_conflict",
            })
            active_collision_tids.update([t1_id, t2_id])

        # -------------------------------------------------------------
        # 3. Detector C: Single-Vehicle Anomaly with Gated Kinematics
        # -------------------------------------------------------------
        for tid, tr in tracks.items():
            if tid in active_collision_tids:
                continue
            if tr.class_label.lower() not in VEHICLES:
                continue
            k = kinematics.get(tid)
            if not k:
                continue

            bbox = tr.current_bbox_xywh
            w, h = bbox[2], bbox[3]
            ar = float(w / max(1.0, h))

            # Edge margin clipping (check all 4 boundaries)
            bx, by = bbox[0], bbox[1]
            if bx < 15.0 or by < 15.0:
                continue

            gx, gy = k.ground_pos
            g_valid = (abs(gx) <= 150.0 and -50.0 <= gy <= 250.0)

            px_spd, px_acc = get_pixel_kinematics(tr, pts_ns)
            char_dim = max(15.0, max(w, h))

            # Scaled kinematics
            if self.is_camera_calibrated and g_valid:
                v_scaled = k.speed_kmh
                a_scaled = k.acceleration_mag
            else:
                v_scaled = (px_spd / char_dim) * 4.5 * 3.6
                a_scaled = (px_acc / char_dim) * 4.5

            if tid not in self.single_track_history:
                self.single_track_history[tid] = []
            self.single_track_history[tid].append({
                "frame_idx": frame_idx,
                "bbox": bbox,
                "ar": ar,
                "heading": k.heading_deg,
                "speed_kmh": v_scaled,
                "accel_ms2": a_scaled,
            })
            hist = self.single_track_history[tid]
            if len(hist) < 4:
                continue
            if len(hist) > 15:
                hist.pop(0)

            prev_entry = hist[-2]
            max_recent_spd = max(h_entry["speed_kmh"] for h_entry in hist[:-1])

            # Signals
            signal_severe_impact = bool(-60.0 <= a_scaled <= -14.0 and max_recent_spd >= 15.0)
            signal_moderate_decel = bool(-14.0 < a_scaled < -4.0)

            dh = abs(k.heading_deg - prev_entry["heading"])
            if dh > 180.0:
                dh = 360.0 - dh
            signal_abnormal_yaw = bool(dh >= 30.0 and max_recent_spd >= 8.0)

            prev_ar = prev_entry["ar"]
            ar_ratio = abs(ar - prev_ar) / max(0.1, prev_ar)
            signal_rollover = bool(ar_ratio >= 0.35)

            # Signal 3: Sudden Stop / Kinetic Energy Loss
            curr_spd = v_scaled
            spd_drop = max_recent_spd - curr_spd
            signal_sudden_stop = bool(max_recent_spd >= 12.0 and curr_spd <= 6.0 and spd_drop >= 10.0)

            is_anomaly = False
            # Solo incident requires abnormal physical kinematics:
            # 1. Rollover (aspect ratio inversion)
            if signal_rollover and max_recent_spd >= 10.0:
                is_anomaly = True
            # 2. Abnormal yaw / spin with deceleration or high speed
            elif signal_abnormal_yaw and (signal_severe_impact or signal_moderate_decel or max_recent_spd >= 20.0):
                is_anomaly = True
            # 3. Severe barrier impact resulting in sudden kinetic arrest (head-on barrier crash)
            elif signal_severe_impact and signal_sudden_stop:
                is_anomaly = True

            if is_anomaly:
                centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
                if self._is_in_lifecycle_incident(centroid, bbox, [tid]):
                    continue

                self.lifecycle_mgr.register_incident(frame_idx, tid, tid, centroid, bbox)
                frame_candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [tid],
                    "dist": 0.0,
                    "iou": 0.0,
                    "accel_1": float(a_scaled),
                    "accel_2": 0.0,
                    "detector": "single_vehicle",
                })

        return frame_candidates


print("Loading cached videos...")
with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    in_dist_vids = pickle.load(f)
with open("cache_videos_iter_b_split_geo_aware_15.pkl", "rb") as f:
    geo_vids = pickle.load(f)

print("Running benchmark evaluation on all 30 videos...")

for split_name, vids in [("in_distribution", in_dist_vids), ("geo_aware", geo_vids)]:
    pipeline = RefinedProductionPipeline(is_camera_calibrated=False)
    total_tp = 0
    total_fn = 0
    total_fp_episodes = 0
    all_cands = 0

    print(f"\n================ Split: {split_name} ================")
    for v in vids:
        vid_id = v["video_id"]
        gt_f = v["gt_frame"]
        fps = v["fps"]
        pipeline.reset()
        raw_cands = []

        for f_idx, pts_ns, tracks, kin, conv in v["frames"]:
            c = pipeline.evaluate_frame(f_idx, pts_ns, tracks, kin, conv)
            raw_cands.extend(c)

        eval_res = evaluate_video_candidates_temporal(
            raw_cands, gt_f, fps, tolerance_frames=35, max_gap_frames=25
        )
        is_tp = eval_res["is_tp"]
        if is_tp:
            total_tp += 1
        else:
            total_fn += 1
        total_fp_episodes += eval_res["fp_episodes"]
        all_cands += eval_res["raw_candidates_count"]
        cand_f = eval_res["matched_episode"]["trigger_frame"] if is_tp else None
        diff_f = (cand_f - gt_f) if cand_f is not None else None

        print(f"Video {vid_id:15s} | GT={gt_f:3d} | {'TP' if is_tp else 'FN'} | Cand={str(cand_f):>4s} (diff={str(diff_f):>4s}) | FP_eps={eval_res['fp_episodes']}")

    prec = total_tp / (total_tp + total_fp_episodes) if (total_tp + total_fp_episodes) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"\n--- Results for {split_name} ---")
    print(f"TP: {total_tp} | FN: {total_fn} | FP Episodes: {total_fp_episodes}")
    print(f"Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1:.3f} | FP/vid: {total_fp_episodes/len(vids):.2f}")
