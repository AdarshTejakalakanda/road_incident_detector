import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)
with open('cache_videos_iter_b_split_geo_aware_15.pkl', 'rb') as f:
    geo = pickle.load(f)

def compute_iou(boxA, boxB):
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 1e-6 else 0.0

class DistantTrajectoryConflictDetector:
    """Detects distant non-overlapping collision candidates using predicted trajectory conflict."""

    VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

    def __init__(
        self,
        is_camera_calibrated: bool = False,
        video_warmup_frames: int = 15,
        min_track_age: int = 3,
        deceleration_threshold_ms2: float = -3.5,
        max_physical_decel_ms2: float = 60.0,
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
        self.tca_max = tca_max_sec
        self.dca_norm_max = dca_norm_max
        self.dca_metric_max = dca_metric_max_m
        self.dist_norm_max = dist_norm_max
        self.dist_metric_max = dist_metric_max_m
        self.min_closing_spd = min_closing_speed_kmh
        self.min_persistence = min_persistence_frames

        self.spatial_clusters: List[Dict[str, Any]] = []
        self.prev_distances: Dict[Tuple[int, int], float] = {}

    def reset(self):
        self.spatial_clusters.clear()
        self.prev_distances.clear()

    def _estimate_metric_acceleration(self, bbox_xywh: Tuple[float, float, float, float], raw_accel: float) -> float:
        if self.is_camera_calibrated:
            return raw_accel
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Any],
        kinematics: Dict[int, Any],
        convergences: List[Any],
        lifecycle_mgr: Optional[Any] = None,
    ) -> List[Dict]:
        candidates = []
        current_pair_keys = set()

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

            if tr1.class_label.lower() not in self.VEHICLES or tr2.class_label.lower() not in self.VEHICLES:
                continue

            # 1. Track Maturity Gate
            if tr1.track_age < self.min_track_age or tr2.track_age < self.min_track_age:
                continue

            # 2. Kinematic Warm-Up Gate
            if len(tr1.ground_history) < 3 or len(tr2.ground_history) < 3:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            dist = conv.distance_m
            prev_d = self.prev_distances.get(pair_key, dist)
            distance_delta = dist - prev_d
            self.prev_distances[pair_key] = dist

            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_dist = dist / char_dim

            # Spatial distance gate (distant perspective envelope)
            if self.is_camera_calibrated:
                within_spatial_envelope = (dist <= self.dist_metric_max)
                predicted_dca_valid = (conv.dca_m <= self.dca_metric_max)
            else:
                within_spatial_envelope = (norm_dist <= self.dist_norm_max)
                predicted_dca_valid = (conv.dca_m / char_dim <= self.dca_norm_max)

            if not within_spatial_envelope:
                continue

            # Trajectory Conflict: closing motion + valid TCA + DCA
            is_closing = (distance_delta < -0.05) or (conv.is_converging and conv.tca_sec < self.tca_max)
            trajectory_conflict = bool(is_closing and conv.is_converging and 0.05 <= conv.tca_sec <= self.tca_max and predicted_dca_valid)

            # Deceleration response in conflict
            raw_a1 = k1.acceleration_mag
            raw_a2 = k2.acceleration_mag
            metric_a1 = self._estimate_metric_acceleration(b1, raw_a1)
            metric_a2 = self._estimate_metric_acceleration(b2, raw_a2)

            valid_decel_1 = (-self.max_physical_decel <= metric_a1 < self.decel_threshold)
            valid_decel_2 = (-self.max_physical_decel <= metric_a2 < self.decel_threshold)
            decel_signal = bool(valid_decel_1 or valid_decel_2)

            # Both moving at meaningful speed prior to deceleration
            relative_speed = np.hypot(k1.velocity_vector[0] - k2.velocity_vector[0], k1.velocity_vector[1] - k2.velocity_vector[1]) * 3.6
            closing_motion = bool(relative_speed >= self.min_closing_spd)

            # Overall distant conflict condition
            is_conflict_candidate = bool(trajectory_conflict and decel_signal and closing_motion)

            if not is_conflict_candidate:
                continue

            # Centroid and interaction bounding box
            centroid = (
                (b1[0] + b1[2]/2.0 + b2[0] + b2[2]/2.0) / 2.0,
                (b1[1] + b1[3]/2.0 + b2[1] + b2[3]/2.0) / 2.0,
            )
            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))

            # Lifecycle check: suppress if already part of an incident region
            if lifecycle_mgr is not None:
                is_in_existing = False
                for inc in lifecycle_mgr.incidents:
                    if inc.state == "CLEARED":
                        continue
                    c_dist = np.hypot(centroid[0] - inc.centroid[0], centroid[1] - inc.centroid[1])
                    spatial_radius = 8.0 if self.is_camera_calibrated else max(120.0, 1.8 * max(inc.bbox[2], inc.bbox[3]))
                    box_iou = compute_iou(interaction_box, inc.bbox)
                    center_inside = (inc.bbox[0] <= centroid[0] <= inc.bbox[0] + inc.bbox[2]) and (inc.bbox[1] <= centroid[1] <= inc.bbox[1] + inc.bbox[3])
                    if c_dist <= spatial_radius or box_iou >= 0.05 or center_inside:
                        is_in_existing = True
                        break
                if is_in_existing:
                    continue

            # Persistence clustering
            matched = False
            triggered = False
            for cluster in self.spatial_clusters:
                if (frame_idx - cluster["last_frame"]) <= 3:
                    box_iou = compute_iou(interaction_box, cluster["box"])
                    c_dist = np.hypot(centroid[0] - cluster["centroid"][0], centroid[1] - cluster["centroid"][1])
                    if box_iou >= 0.05 or c_dist <= 25.0:
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
                if lifecycle_mgr is not None:
                    lifecycle_mgr.register_incident(
                        frame_idx, t1_id, t2_id, centroid, interaction_box
                    )
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [t1_id, t2_id],
                    "dist": float(dist),
                    "iou": float(compute_iou(b1, b2)),
                    "accel_1": float(metric_a1),
                    "accel_2": float(metric_a2),
                    "tca_sec": float(conv.tca_sec),
                    "dca_m": float(conv.dca_m),
                    "detector": "distant_conflict",
                })

        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]

        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates

from scripts.benchmark_iteration_b_detector_recovery import (
    cluster_episodes,
    evaluate_video_candidates_temporal,
    RestoredIncidentLifecycleManager,
)

for split_name, dataset in [("split_in_distribution", in_dist), ("split_geo_aware", geo)]:
    print(f"\n==========================================")
    print(f"EVALUATING DISTANT CONFLICT DETECTOR ON {split_name}")
    print(f"==========================================")
    det = DistantTrajectoryConflictDetector(is_camera_calibrated=False)
    lm = RestoredIncidentLifecycleManager(is_camera_calibrated=False)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    video_reports = []
    
    for v in dataset:
        det.reset()
        lm.reset()
        vid_cands = []
        
        for f_idx, pts, tracks, kin, conv in v["frames"]:
            cands = det.evaluate_frame(f_idx, pts, tracks, kin, conv, lm)
            vid_cands.extend(cands)
            
        res = evaluate_video_candidates_temporal(
            vid_cands, v["gt_frame"], v["fps"], tolerance_frames=35, max_gap_frames=25
        )
        
        if res["is_tp"]:
            total_tp += 1
            status = "TP"
        else:
            total_fn += 1
            status = "FN"
        total_fp += res["fp_episodes"]
        
        video_reports.append((v["video_id"], status, v["gt_frame"], [c["frame_idx"] for c in vid_cands], res["fp_episodes"]))
        
    prec = total_tp / max(1, total_tp + total_fp)
    rec = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    
    print(f"Results for Distant Conflict Detector standalone on {split_name}:")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("Videos breakdown:")
    for vid, status, gt, c_frames, fp_count in video_reports:
        print(f"  {vid:18s} | {status} | GT={gt:3d} | cands={c_frames} | FP_ep={fp_count}")
