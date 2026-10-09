"""Iteration A: Calibration-Aware Collision Geometry Benchmark Runner.

Implements Iteration A:
if camera_calibrated:
    use ground-distance + IoU + convergence
else:
    use 2D IoU / image-space collision geometry
    do NOT interpret pixel separation as meters

Combines with:
- Stage 6 spatial continuity (>= 2 frames)
- Track maturity gate (track_age >= 3)
- Kinematic warm-up (>= 3 history frames)
- Physically plausible acceleration bounds (-60 to -3.5 m/s2 metric)
- Explicit Incident Region & Lifecycle State Management
- Episode-level deduplication (Delta f <= 25 frames)

Evaluates on BOTH:
- split_in_distribution (15 test videos)
- split_geo_aware (15 test videos)
"""

import os
import sys
import json
import pickle
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config, CollisionConfig
from road_surveillance.tracking import Track
from road_surveillance.calibration import PlanarHomography, LaneManager
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from road_surveillance.events.candidates.collision import CollisionCandidateDetector


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute 2D bounding box Intersection over Union."""
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 1e-6 else 0.0


class IncidentRegion:
    """Spatial and state representation of an active incident hazard zone."""

    def __init__(
        self,
        region_id: int,
        origin_frame: int,
        centroid: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        involved_track_ids: Set[int],
    ):
        self.region_id = region_id
        self.origin_frame = origin_frame
        self.centroid = centroid
        self.bbox = bbox
        self.involved_track_ids = set(involved_track_ids)
        self.resting_track_ids: Set[int] = set()
        self.state = "ACTIVE_COLLISION"
        self.last_active_frame = origin_frame

    def update_bounds(self, new_bbox: Tuple[float, float, float, float]):
        bx, by, bw, bh = self.bbox
        nx, ny, nw, nh = new_bbox
        ux1 = min(bx, nx)
        uy1 = min(by, ny)
        ux2 = max(bx + bw, nx + nw)
        uy2 = max(by + bh, ny + nh)
        self.bbox = (ux1, uy1, ux2 - ux1, uy2 - uy1)


class IncidentLifecycleManager:
    """Tracks active incident regions and filters redundant resting wreckage alarms."""

    def __init__(self, spatial_radius_m: float = 8.0):
        self.spatial_radius_m = spatial_radius_m
        self.incidents: List[IncidentRegion] = []
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
        centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[str]]:
        for inc in self.incidents:
            if inc.state == "CLEARED":
                continue

            c_dist = np.hypot(centroid[0] - inc.centroid[0], centroid[1] - inc.centroid[1])
            box_iou = compute_iou(interaction_bbox, inc.bbox)

            # Within incident spatial hazard zone
            if c_dist <= self.spatial_radius_m or box_iou >= 0.05:
                t1_in_inc = (t1_id in inc.involved_track_ids) or (t1_id in inc.resting_track_ids)
                t2_in_inc = (t2_id in inc.involved_track_ids) or (t2_id in inc.resting_track_ids)
                s1_low = k1.speed_kmh < 6.0
                s2_low = k2.speed_kmh < 6.0

                # 1. Damaged / Resting Wreckage Jitter
                if (t1_in_inc and t2_in_inc) or (s1_low and s2_low):
                    inc.resting_track_ids.add(t1_id)
                    inc.resting_track_ids.add(t2_id)
                    inc.state = "POST_IMPACT_RESTING"
                    inc.last_active_frame = frame_idx
                    inc.update_bounds(interaction_bbox)
                    return True, "WRECKAGE_RESTING"

                # 2. Passing Vehicle Grazing Stationary Wreckage
                if (t1_in_inc and not t2_in_inc) or (t2_in_inc and not t1_in_inc):
                    moving_k = k2 if t1_in_inc else k1
                    if moving_k.acceleration_mag >= -3.5 or moving_k.speed_kmh > 8.0:
                        inc.last_active_frame = frame_idx
                        return True, "PASSING_VEHICLE_RESTING_INTERACTION"

        return False, None

    def register_incident(
        self,
        frame_idx: int,
        t1_id: int,
        t2_id: int,
        centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ):
        bx, by, bw, bh = interaction_bbox
        mw = bw * 0.3
        mh = bh * 0.3
        expanded = (max(0.0, bx - mw), max(0.0, by - mh), bw + 2 * mw, bh + 2 * mh)

        inc = IncidentRegion(
            region_id=self.next_id,
            origin_frame=frame_idx,
            centroid=centroid,
            bbox=expanded,
            involved_track_ids={t1_id, t2_id},
        )
        self.incidents.append(inc)
        self.next_id += 1


class CalibrationAwareDetector:
    """Collision Detector supporting both metric-calibrated and image-space uncalibrated modes."""

    def __init__(
        self,
        mode: str,
        is_camera_calibrated: bool = False,
        config: Optional[CollisionConfig] = None,
        max_physical_decel_ms2: float = 60.0,
    ):
        self.mode = mode
        self.is_camera_calibrated = is_camera_calibrated
        self.cfg = config or CollisionConfig()
        self.max_physical_decel = max_physical_decel_ms2
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_ious: Dict[Tuple[int, int], float] = {}
        self.spatial_clusters: List[Dict[str, Any]] = []
        self.lifecycle_mgr = IncidentLifecycleManager()

    def reset(self):
        self.persistence_counts.clear()
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.spatial_clusters.clear()
        self.lifecycle_mgr.reset()

    def _estimate_metric_acceleration(self, track: Track, raw_accel: float) -> float:
        if self.is_camera_calibrated:
            return raw_accel
        w, h = track.current_bbox_xywh[2], track.current_bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
    ) -> List[Dict]:
        candidates = []
        current_pair_keys = set()
        VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

        for conv in convergences:
            t1_id = conv.track_id_1
            t2_id = conv.track_id_2
            k1 = kinematics.get(t1_id)
            k2 = kinematics.get(t2_id)
            tr1 = tracks.get(t1_id)
            tr2 = tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            dist = conv.distance_m
            prev_d = self.prev_distances.get(pair_key, dist)
            distance_delta = dist - prev_d
            is_closing = (distance_delta < -0.1)
            self.prev_distances[pair_key] = dist

            iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            # 1. Track Maturity Gate
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            # 2. Kinematic Warm-Up (>= 3 history frames)
            warm1 = len(tr1.ground_history) >= 3 and len(tr1.timestamps) >= 3
            warm2 = len(tr2.ground_history) >= 3 and len(tr2.timestamps) >= 3

            raw_a1 = k1.acceleration_mag if warm1 else 0.0
            raw_a2 = k2.acceleration_mag if warm2 else 0.0

            # 3. Physically Plausible Deceleration Bounds
            metric_a1 = self._estimate_metric_acceleration(tr1, raw_a1)
            metric_a2 = self._estimate_metric_acceleration(tr2, raw_a2)

            valid_decel_1 = (-self.max_physical_decel <= metric_a1 < self.cfg.deceleration_threshold_ms2)
            valid_decel_2 = (-self.max_physical_decel <= metric_a2 < self.cfg.deceleration_threshold_ms2)
            decel_signal = bool(valid_decel_1 or valid_decel_2)

            # 4. Calibration-Aware Proximity & Convergence Gating
            if self.mode == "previous_best_fixed_metric":
                # Old rigid metric distance check (interprets uncalibrated pixels as meters)
                proximity_signal = (dist <= self.cfg.max_ground_distance_m) or (
                    iou >= self.cfg.min_iou_overlap and dist <= self.cfg.max_perspective_distance_m
                )
                conv_signal = bool(conv.is_converging and conv.tca_sec < self.cfg.tca_threshold_sec and conv.dca_m < self.cfg.post_proximity_distance_m)
            else:
                # Iteration A: Calibration-Aware
                if self.is_camera_calibrated:
                    # Calibrated camera: ground plane distance (meters) + IoU + trajectory convergence
                    proximity_signal = (dist <= self.cfg.max_ground_distance_m) or (
                        iou >= self.cfg.min_iou_overlap and dist <= self.cfg.max_perspective_distance_m
                    )
                    conv_signal = bool(conv.is_converging and conv.tca_sec < self.cfg.tca_threshold_sec and conv.dca_m < self.cfg.post_proximity_distance_m)
                else:
                    # Uncalibrated camera: 2D image-space collision geometry
                    # Do NOT interpret pixel separation as meters!
                    char_dim = max(15.0, (max(tr1.current_bbox_xywh[2], tr1.current_bbox_xywh[3]) + max(tr2.current_bbox_xywh[2], tr2.current_bbox_xywh[3])) / 2.0)
                    norm_pixel_dist = dist / char_dim
                    # Proximity satisfied if 2D IoU overlap >= 0.15 OR bottom contact centers within 0.8 vehicle lengths
                    proximity_signal = (iou >= self.cfg.min_iou_overlap) or (norm_pixel_dist <= 0.8)
                    # Convergence satisfied if distance closing OR IoU expanding OR metric TCA < 2s
                    conv_signal = bool(is_closing or iou_increasing or (conv.is_converging and conv.tca_sec < self.cfg.tca_threshold_sec))

            centroid = (
                (k1.ground_pos[0] + k2.ground_pos[0]) / 2.0,
                (k1.ground_pos[1] + k2.ground_pos[1]) / 2.0,
            )
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))

            # 5. Incident Lifecycle Gating
            is_redundant, reason = self.lifecycle_mgr.evaluate_candidate_interaction(
                frame_idx, t1_id, t2_id, tr1, tr2, k1, k2, centroid, interaction_box
            )
            if is_redundant:
                continue

            cond = proximity_signal and conv_signal and decel_signal
            triggered = False

            if cond:
                matched = False
                for cluster in self.spatial_clusters:
                    if cluster["last_frame"] < frame_idx and (frame_idx - cluster["last_frame"]) <= 3:
                        box_iou = compute_iou(interaction_box, cluster["box"])
                        c_dist = np.hypot(centroid[0] - cluster["centroid"][0], centroid[1] - cluster["centroid"][1])
                        if box_iou >= 0.10 or c_dist <= 5.0:
                            cluster["persistence"] += 1
                            cluster["last_frame"] = frame_idx
                            cluster["centroid"] = centroid
                            cluster["box"] = interaction_box
                            matched = True
                            if cluster["persistence"] >= self.cfg.min_persistence_frames:
                                triggered = True
                            break
                if not matched:
                    self.spatial_clusters.append({
                        "centroid": centroid,
                        "box": interaction_box,
                        "last_frame": frame_idx,
                        "persistence": 1,
                    })
                    if self.cfg.min_persistence_frames <= 1:
                        triggered = True

            if triggered:
                self.lifecycle_mgr.register_incident(
                    frame_idx, t1_id, t2_id, centroid, interaction_box
                )
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [t1_id, t2_id],
                    "dist": float(dist),
                    "iou": float(iou),
                    "accel_1": float(k1.acceleration_mag),
                    "accel_2": float(k2.acceleration_mag),
                })

        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
                self.prev_ious.pop(pk, None)
                self.persistence_counts.pop(pk, None)

        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates


def cluster_episodes(raw_candidates: List[Dict], max_gap_frames: int = 25) -> List[Dict]:
    if not raw_candidates:
        return []

    sorted_cands = sorted(raw_candidates, key=lambda c: c["frame_idx"])
    episodes = []
    current_ep = [sorted_cands[0]]

    for c in sorted_cands[1:]:
        if (c["frame_idx"] - current_ep[-1]["frame_idx"]) <= max_gap_frames:
            current_ep.append(c)
        else:
            rep = current_ep[0]
            episodes.append({
                "start_frame": current_ep[0]["frame_idx"],
                "end_frame": current_ep[-1]["frame_idx"],
                "trigger_frame": current_ep[0]["frame_idx"],
                "raw_count": len(current_ep),
                "representative": rep,
            })
            current_ep = [c]

    if current_ep:
        rep = current_ep[0]
        episodes.append({
            "start_frame": current_ep[0]["frame_idx"],
            "end_frame": current_ep[-1]["frame_idx"],
            "trigger_frame": current_ep[0]["frame_idx"],
            "raw_count": len(current_ep),
            "representative": rep,
        })

    return episodes


def evaluate_video_candidates_temporal(
    raw_candidates: List[Dict],
    gt_frame: int,
    fps: float,
    tolerance_frames: int = 35,
    max_gap_frames: int = 25,
) -> Dict[str, Any]:
    episodes = cluster_episodes(raw_candidates, max_gap_frames=max_gap_frames)

    matched_episode = None
    pre_impact_fps = []
    post_impact_fps = []
    repeated_incident_fps = []
    raw_fp_count = 0

    for ep in episodes:
        tf = ep["trigger_frame"]
        diff = tf - gt_frame

        if abs(diff) <= tolerance_frames and matched_episode is None:
            matched_episode = ep
        else:
            raw_fp_count += ep["raw_count"]
            if diff < -tolerance_frames:
                pre_impact_fps.append(ep)
            else:
                post_impact_fps.append(ep)
                if diff > tolerance_frames and abs(diff) <= 800:
                    repeated_incident_fps.append(ep)

    is_tp = matched_episode is not None
    lead_frames = (gt_frame - matched_episode["trigger_frame"]) if is_tp else None
    lead_sec = (lead_frames / fps) if (is_tp and fps > 0) else None

    return {
        "is_tp": is_tp,
        "matched_episode": matched_episode,
        "lead_frames": lead_frames,
        "lead_sec": lead_sec,
        "total_episodes": len(episodes),
        "fp_episodes": len(pre_impact_fps) + len(post_impact_fps),
        "pre_impact_fp_count": len(pre_impact_fps),
        "post_impact_fp_count": len(post_impact_fps),
        "repeated_incident_fp_count": len(repeated_incident_fps),
        "raw_fp_count": raw_fp_count,
        "raw_candidates_count": len(raw_candidates),
    }


def run_benchmark():
    cfg = load_config("road_surveillance/config/default_config.yaml")

    exp_cfg = CollisionConfig(
        max_ground_distance_m=4.0,
        max_perspective_distance_m=12.0,
        min_iou_overlap=0.15,
        min_persistence_frames=2,
        temporal_cluster_gap_frames=25,
    )

    baseline_engine = CollisionCandidateDetector(
        distance_decay_threshold_m=cfg.events.collision.distance_decay_threshold_m,
        tca_threshold_sec=cfg.events.collision.tca_threshold_sec,
        deceleration_threshold_ms2=cfg.events.collision.deceleration_threshold_ms2,
        post_proximity_distance_m=cfg.events.collision.post_proximity_distance_m,
        iou_overlap_threshold=cfg.events.collision.iou_overlap_threshold,
    )

    det_prev_best = CalibrationAwareDetector(mode="previous_best_fixed_metric", is_camera_calibrated=False, config=exp_cfg)
    det_iteration_a = CalibrationAwareDetector(mode="iteration_a_calibration_aware", is_camera_calibrated=False, config=exp_cfg)

    configs = [
        ("0_baseline", "Baseline (Untouched Multi-Signal)"),
        ("1_prev_best", "Previous Best (Fixed Metric Distance Gate)"),
        ("2_iteration_a", "Iteration A (Calibration-Aware Collision Geometry)"),
    ]

    all_split_results = {}

    for split_type in ["split_in_distribution", "split_geo_aware"]:
        cache_file = f"cache_videos_{split_type}_15.pkl"
        with open(cache_file, "rb") as f:
            vids = pickle.load(f)

        print(f"\nEvaluating {len(configs)} configurations on '{split_type}' ({len(vids)} cached videos)...")
        results = {}

        for cid, clabel in configs:
            tp_count = 0
            fn_count = 0
            fp_episodes_total = 0
            raw_fp_total = 0
            pre_impact_fps_total = 0
            post_impact_fps_total = 0
            repeated_fps_total = 0
            lead_times_frames = []
            lead_times_sec = []
            abs_temp_errors = []

            for v in vids:
                gt_f = v["gt_frame"]
                fps = v["fps"]
                vid_id = v["video_id"]

                if cid == "0_baseline":
                    baseline_engine.prev_distances.clear()
                    cands = []
                    for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                        raw_c = baseline_engine.evaluate(f_idx, pts_ns, vid_id, tr, kin, conv, LaneManager())
                        for rc in raw_c:
                            cands.append({
                                "frame_idx": rc.frame_idx,
                                "pts_ns": rc.timestamp_ns,
                                "track_ids": rc.track_ids,
                                "dist": rc.signals.get("ground_distance_m", 0.0),
                                "iou": rc.signals.get("bbox_iou", 0.0),
                                "accel_1": rc.signals.get("accel_1_ms2", 0.0),
                                "accel_2": rc.signals.get("accel_2_ms2", 0.0),
                            })
                elif cid == "1_prev_best":
                    det_prev_best.reset()
                    cands = []
                    for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                        cands.extend(det_prev_best.evaluate_frame(f_idx, pts_ns, tr, kin, conv))
                else:
                    det_iteration_a.reset()
                    cands = []
                    for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                        cands.extend(det_iteration_a.evaluate_frame(f_idx, pts_ns, tr, kin, conv))

                eval_res = evaluate_video_candidates_temporal(
                    cands, gt_f, fps, tolerance_frames=35, max_gap_frames=25
                )

                if eval_res["is_tp"]:
                    tp_count += 1
                    lead_times_frames.append(eval_res["lead_frames"])
                    lead_times_sec.append(eval_res["lead_sec"])
                    abs_temp_errors.append(abs(eval_res["lead_frames"]))
                else:
                    fn_count += 1

                fp_episodes_total += eval_res["fp_episodes"]
                raw_fp_total += eval_res["raw_fp_count"]
                pre_impact_fps_total += eval_res["pre_impact_fp_count"]
                post_impact_fps_total += eval_res["post_impact_fp_count"]
                repeated_fps_total += eval_res["repeated_incident_fp_count"]

            n_vids = len(vids)
            prec = tp_count / (tp_count + fp_episodes_total) if (tp_count + fp_episodes_total) > 0 else 0.0
            rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            results[cid] = {
                "label": clabel,
                "tp": tp_count,
                "fp_episodes": fp_episodes_total,
                "raw_fp": raw_fp_total,
                "fn": fn_count,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fp_per_vid": fp_episodes_total / n_vids,
                "raw_fp_per_vid": raw_fp_total / n_vids,
                "pre_impact_fp": pre_impact_fps_total,
                "post_impact_fp": post_impact_fps_total,
                "repeated_fp": repeated_fps_total,
                "med_lead_sec": float(np.median(lead_times_sec)) if lead_times_sec else 0.0,
                "med_lead_frames": float(np.median(lead_times_frames)) if lead_times_frames else 0.0,
                "med_abs_error": float(np.median(abs_temp_errors)) if abs_temp_errors else 0.0,
            }

        all_split_results[split_type] = results

    # Print & Save Report
    lines = [
        "# Iteration A: Calibration-Aware Collision Geometry Benchmark Report",
        "**Evaluated Splits**: `split_in_distribution` (15 videos) vs `split_geo_aware` (15 videos) | **GT Window**: $\\pm 35$ frames (~1.17s)",
        "",
        "## 1. Primary Incident Detection Metrics",
    ]

    for split_type in ["split_in_distribution", "split_geo_aware"]:
        lines.extend([
            f"### Split: `{split_type}`",
            "| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cid, r in all_split_results[split_type].items():
            lines.append(
                f"| **{r['label']}** | {r['tp']} | {r['fp_episodes']} | {r['fn']} | "
                f"**{r['precision']:.3f}** | **{r['recall']:.3f}** | **{r['f1']:.3f}** | "
                f"{r['fp_per_vid']:.2f} | {r['raw_fp_per_vid']:.1f} | "
                f"+{r['med_lead_sec']:.2f}s ({r['med_lead_frames']:+.0f}f) | {r['med_abs_error']:.0f} frames |"
            )
        lines.append("")

    lines.extend([
        "## 2. Temporal False Positive Taxonomy Breakdown",
        "| Split | Configuration | Pre-Impact FP | Post-Impact FP | Repeated Incident FP | Total FP Episodes |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for split_type, s_name in [("split_in_distribution", "In-Dist"), ("split_geo_aware", "Geo-Aware")]:
        for cid, r in all_split_results[split_type].items():
            lines.append(
                f"| **{s_name}** | **{r['label']}** | "
                f"{r['pre_impact_fp']} | {r['post_impact_fp']} | {r['repeated_fp']} | **{r['fp_episodes']}** |"
            )

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    with open("docs/iteration_a_calibration_aware_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Report saved to: docs/iteration_a_calibration_aware_report.md")


if __name__ == "__main__":
    run_benchmark()
