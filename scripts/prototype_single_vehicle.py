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

class SingleVehicleAnomalyDetector:
    VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

    def __init__(
        self,
        is_camera_calibrated: bool = False,
        video_warmup_frames: int = 15,
        min_track_age: int = 6,
        deceleration_threshold_ms2: float = -4.0,
        severe_impact_decel_ms2: float = -14.0,
        max_physical_decel_ms2: float = 60.0,
        yaw_rate_threshold_deg: float = 30.0,
        aspect_ratio_change_threshold: float = 0.35,
        sudden_stop_speed_drop_kmh: float = 15.0,
        min_persistence_frames: int = 2,
    ):
        self.is_camera_calibrated = is_camera_calibrated
        self.video_warmup_frames = video_warmup_frames
        self.min_track_age = min_track_age
        self.decel_threshold = deceleration_threshold_ms2
        self.severe_impact_decel = severe_impact_decel_ms2
        self.max_physical_decel = max_physical_decel_ms2
        self.yaw_rate_thresh = yaw_rate_threshold_deg
        self.ar_change_thresh = aspect_ratio_change_threshold
        self.sudden_stop_drop = sudden_stop_speed_drop_kmh
        self.min_persistence = min_persistence_frames

        self.track_history: Dict[int, List[Dict[str, Any]]] = {}
        self.spatial_clusters: List[Dict[str, Any]] = []

    def reset(self):
        self.track_history.clear()
        self.spatial_clusters.clear()

    def _estimate_metric_acceleration(self, bbox_xywh: Tuple[float, float, float, float], raw_accel: float) -> float:
        if self.is_camera_calibrated:
            return raw_accel
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

    def _estimate_metric_speed(self, bbox_xywh: Tuple[float, float, float, float], raw_speed: float) -> float:
        if self.is_camera_calibrated:
            return raw_speed
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_speed * scale

    def _is_edge_clipped(self, bbox_xywh: Tuple[float, float, float, float], margin: float = 20.0) -> bool:
        bx, by, bw, bh = bbox_xywh
        if bx < margin or by < margin:
            return True
        return False

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Any],
        kinematics: Dict[int, Any],
        lifecycle_mgr: Optional[Any] = None,
    ) -> List[Dict]:
        candidates = []
        active_tids = set()

        if frame_idx < self.video_warmup_frames:
            return candidates

        for tid, tr in tracks.items():
            if tr.class_label.lower() not in self.VEHICLES:
                continue

            k = kinematics.get(tid)
            if not k:
                continue

            active_tids.add(tid)
            bbox = tr.current_bbox_xywh
            w, h = bbox[2], bbox[3]
            ar = float(w / max(1.0, h))

            raw_a = k.acceleration_mag
            metric_a = self._estimate_metric_acceleration(bbox, raw_a)
            metric_speed = self._estimate_metric_speed(bbox, k.speed_kmh)

            entry = {
                "frame_idx": frame_idx,
                "bbox": bbox,
                "ar": ar,
                "heading": k.heading_deg,
                "speed_kmh": metric_speed,
                "accel_ms2": metric_a,
                "ground_pos": k.ground_pos,
                "track_age": tr.track_age,
            }

            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append(entry)
            hist = self.track_history[tid]

            # 1. Track Maturity Gate
            if tr.track_age < self.min_track_age or len(hist) < self.min_track_age or len(tr.ground_history) < self.min_track_age:
                continue

            # Reject edge-clipped boundary tracks
            if self._is_edge_clipped(bbox):
                continue

            if len(hist) > 15:
                hist.pop(0)

            prev_entry = hist[-2]

            # 2. Physically plausible bounds
            if abs(metric_a) > self.max_physical_decel:
                continue

            # Signal 1: Severe barrier impact deceleration with speed history
            max_recent_spd = max(h["speed_kmh"] for h in hist[:-1])
            signal_severe_impact = bool(-self.max_physical_decel <= metric_a <= self.severe_impact_decel and max_recent_spd >= 15.0)

            # Signal 2: Moderate Deceleration
            signal_moderate_decel = bool(self.severe_impact_decel < metric_a < self.decel_threshold)

            # Signal 3: Sudden Stop / Kinetic Energy Loss
            curr_spd = metric_speed
            spd_drop = max_recent_spd - curr_spd
            signal_sudden_stop = bool(max_recent_spd >= 12.0 and curr_spd <= 4.0 and spd_drop >= self.sudden_stop_drop)

            # Signal 4: Abnormal Yaw / Heading Deviation
            dh = abs(entry["heading"] - prev_entry["heading"])
            if dh > 180.0:
                dh = 360.0 - dh
            signal_abnormal_yaw = bool(dh >= self.yaw_rate_thresh and max_recent_spd >= 8.0)

            # Signal 5: Rollover / Aspect Ratio Inversion
            prev_ar = prev_entry["ar"]
            ar_ratio = abs(ar - prev_ar) / max(0.1, prev_ar)
            signal_rollover = bool(ar_ratio >= self.ar_change_thresh)

            # Multi-signal anomaly logic:
            is_anomaly = False
            immediate_trigger = False
            active_signals = []

            if signal_severe_impact:
                is_anomaly = True
                immediate_trigger = True
                active_signals.append("severe_barrier_impact")
            elif signal_moderate_decel and (signal_abnormal_yaw or signal_rollover):
                is_anomaly = True
                active_signals.extend(["decel", "yaw" if signal_abnormal_yaw else "rollover"])
            elif signal_sudden_stop and (signal_abnormal_yaw or signal_rollover):
                is_anomaly = True
                active_signals.extend(["stop", "yaw" if signal_abnormal_yaw else "rollover"])
            elif signal_rollover and (signal_abnormal_yaw or signal_moderate_decel or signal_sudden_stop):
                is_anomaly = True
                active_signals.extend(["rollover", "motion_anomaly"])

            if not is_anomaly:
                continue

            centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
            if lifecycle_mgr is not None:
                is_in_existing = False
                for inc in lifecycle_mgr.incidents:
                    if inc.state == "CLEARED":
                        continue
                    c_dist = np.hypot(centroid[0] - inc.centroid[0], centroid[1] - inc.centroid[1])
                    spatial_radius = 8.0 if self.is_camera_calibrated else max(120.0, 1.8 * max(inc.bbox[2], inc.bbox[3]))
                    box_iou = compute_iou(bbox, inc.bbox)
                    if c_dist <= spatial_radius or box_iou >= 0.05 or tid in inc.involved_track_ids or tid in inc.resting_track_ids:
                        is_in_existing = True
                        break
                if is_in_existing:
                    continue

            matched = False
            triggered = immediate_trigger
            if not triggered:
                for cluster in self.spatial_clusters:
                    if cluster["track_id"] == tid and (frame_idx - cluster["last_frame"]) <= 3:
                        cluster["persistence"] += 1
                        cluster["last_frame"] = frame_idx
                        cluster["box"] = bbox
                        matched = True
                        if cluster["persistence"] >= self.min_persistence:
                            triggered = True
                        break

                if not matched:
                    self.spatial_clusters.append({
                        "track_id": tid,
                        "centroid": centroid,
                        "box": bbox,
                        "last_frame": frame_idx,
                        "persistence": 1,
                    })
                    if self.min_persistence <= 1:
                        triggered = True

            if triggered:
                if lifecycle_mgr is not None:
                    lifecycle_mgr.register_incident(
                        frame_idx, tid, tid, centroid, bbox
                    )
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [tid],
                    "dist": 0.0,
                    "iou": 0.0,
                    "accel_1": float(metric_a),
                    "accel_2": 0.0,
                    "anomaly_type": "+".join(active_signals),
                })

        for t in list(self.track_history.keys()):
            if t not in active_tids:
                del self.track_history[t]

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
    det = SingleVehicleAnomalyDetector(is_camera_calibrated=False, video_warmup_frames=15, min_track_age=6, min_persistence_frames=2)
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
            cands = det.evaluate_frame(f_idx, pts, tracks, kin, lm)
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
    
    print(f"\nResults for Single-Vehicle Detector standalone on {split_name}:")
    print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"  Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("Videos breakdown:")
    for vid, status, gt, c_frames, fp_count in video_reports:
        print(f"  {vid:18s} | {status} | GT={gt:3d} | cands={c_frames} | FP_ep={fp_count}")
