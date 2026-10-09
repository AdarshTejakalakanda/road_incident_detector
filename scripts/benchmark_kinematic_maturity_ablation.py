"""Kinematic Maturity Gate, Warm-up, and Physical Acceleration Bounds Ablation Runner.

Compares:
1. Baseline (Untouched Multi-Signal Detector)
2. Stage 6 (Spatial Continuity Track-Switch Robustness)
3. Stage 6 + Track Maturity Gate (track_age >= 3)
4. Stage 6 + Maturity + Kinematic Validity (warm-up >= 3 history frames + physical acceleration bounds)

Evaluates on BOTH:
- split_in_distribution
- split_geo_aware
"""

import os
import sys
import json
import pickle
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config, CollisionConfig
from road_surveillance.video import DatasetVideoIngestor
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import ByteTrackTracker, Track
from road_surveillance.calibration import PlanarHomography, LaneManager
from road_surveillance.geometry import GeometryEngine
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from road_surveillance.events.candidates.collision import CollisionCandidateDetector
from road_surveillance.evaluation import GroundTruthLoader


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


def classify_false_positive(
    tr1: Track,
    tr2: Track,
    k1: TrackKinematics,
    k2: TrackKinematics,
    conv: TrajectoryConvergence,
    dist: float,
    iou: float,
) -> str:
    """Classify a false positive episode into one of 7 standardized failure modes."""
    # 1. Tracker Artifact
    if abs(k1.acceleration_mag) > 25.0 or abs(k2.acceleration_mag) > 25.0:
        return "tracker_artifact"
    if tr1.track_age <= 3 or tr2.track_age <= 3:
        return "tracker_artifact"

    # 2. Calibration / Homography Artifact
    g1, g2 = k1.ground_pos, k2.ground_pos
    if abs(g1[0]) > 35.0 or abs(g2[0]) > 35.0 or g1[1] > 90.0 or g2[1] > 90.0 or g1[1] < -10.0 or g2[1] < -10.0:
        return "calibration_artifact"
    p1 = (tr1.current_bbox_xywh[0] + tr1.current_bbox_xywh[2] / 2.0, tr1.current_bbox_xywh[1] + tr1.current_bbox_xywh[3] / 2.0)
    p2 = (tr2.current_bbox_xywh[0] + tr2.current_bbox_xywh[2] / 2.0, tr2.current_bbox_xywh[1] + tr2.current_bbox_xywh[3] / 2.0)
    pixel_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
    if dist < 2.0 and iou < 0.02 and pixel_dist > 150.0:
        return "calibration_artifact"

    # 3. Stopped Traffic
    if k1.speed_kmh < 4.0 or k2.speed_kmh < 4.0:
        return "stopped_traffic"

    # Directional analysis
    v1 = np.array(k1.velocity_vector)
    v2 = np.array(k2.velocity_vector)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    cos_theta = float(np.dot(v1, v2) / (norm1 * norm2)) if (norm1 > 1e-3 and norm2 > 1e-3) else 1.0

    # 4. Crossing Traffic
    if abs(cos_theta) < 0.45 and k1.speed_kmh > 8.0 and k2.speed_kmh > 8.0:
        return "crossing_traffic"

    # 5. Normal Following
    if cos_theta > 0.75:
        rel_speed = abs(k1.speed_kmh - k2.speed_kmh)
        if rel_speed < 20.0 and k1.acceleration_mag > -3.0 and k2.acceleration_mag > -3.0:
            return "normal_following"

    # 6. Lane Merge
    if 0.45 <= cos_theta <= 0.85 and conv.is_converging:
        return "lane_merge"

    return "other"


class KinematicMaturityDetector:
    """Evaluates Stage 6 variants with Track Maturity, Warm-up, and Physical Acceleration Bounds."""

    def __init__(
        self,
        mode: str,
        config: Optional[CollisionConfig] = None,
        max_physical_decel_ms2: float = 60.0,
    ):
        self.mode = mode
        self.cfg = config or CollisionConfig()
        self.max_physical_decel = max_physical_decel_ms2
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.spatial_clusters: List[Dict[str, Any]] = []

    def reset(self):
        self.persistence_counts.clear()
        self.prev_distances.clear()
        self.spatial_clusters.clear()

    def _estimate_metric_acceleration(self, track: Track, raw_accel: float) -> float:
        """Estimate metric acceleration accounting for uncalibrated pixel scale."""
        # Standard passenger car length ~ 4.5m
        w, h = track.current_bbox_xywh[2], track.current_bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale_m_per_px = 4.5 / char_px
        return raw_accel * scale_m_per_px

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

            # -------------------------------------------------------------
            # Feature 1: Track Maturity Gate
            # -------------------------------------------------------------
            if self.mode in ("stage6_maturity", "stage6_maturity_validity"):
                if tr1.track_age < 3 or tr2.track_age < 3:
                    # Ignore immature tracks completely
                    continue

            # -------------------------------------------------------------
            # Feature 2 & 3: Kinematic Warm-up & Physically Plausible Bounds
            # -------------------------------------------------------------
            if self.mode == "stage6_maturity_validity":
                # Warm-up check: acceleration is only valid if track has >= 3 ground history positions
                # (prevents calculating acceleration from a single displacement from 0)
                warm1 = len(tr1.ground_history) >= 3 and len(tr1.timestamps) >= 3
                warm2 = len(tr2.ground_history) >= 3 and len(tr2.timestamps) >= 3

                raw_a1 = k1.acceleration_mag if warm1 else 0.0
                raw_a2 = k2.acceleration_mag if warm2 else 0.0

                # Metric-scaled acceleration for physical plausibility check
                metric_a1 = self._estimate_metric_acceleration(tr1, raw_a1)
                metric_a2 = self._estimate_metric_acceleration(tr2, raw_a2)

                # Plausible deceleration: must be steeper than threshold (-3.5 m/s2)
                # but NOT unphysical sensor noise (> max_physical_decel, e.g. -60 m/s2 / ~6g)
                valid_decel_1 = (-self.max_physical_decel <= metric_a1 < self.cfg.deceleration_threshold_ms2)
                valid_decel_2 = (-self.max_physical_decel <= metric_a2 < self.cfg.deceleration_threshold_ms2)

                decel_signal = bool(valid_decel_1 or valid_decel_2)
            else:
                # Standard Stage 6 decel check (no warm-up, raw threshold)
                decel_signal = bool(k1.acceleration_mag < self.cfg.deceleration_threshold_ms2 or k2.acceleration_mag < self.cfg.deceleration_threshold_ms2)

            iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            conv_signal = bool(conv.is_converging and conv.tca_sec < self.cfg.tca_threshold_sec and conv.dca_m < self.cfg.post_proximity_distance_m)
            strict_proximity = (dist <= self.cfg.max_ground_distance_m) or (iou >= self.cfg.min_iou_overlap and dist <= self.cfg.max_perspective_distance_m)

            # Spatial continuity condition
            cond = strict_proximity and (conv_signal or is_closing) and decel_signal
            triggered = False

            if cond:
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
                fp_cat = classify_false_positive(tr1, tr2, k1, k2, conv, dist, iou)
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [t1_id, t2_id],
                    "dist": float(dist),
                    "iou": float(iou),
                    "is_converging": bool(conv.is_converging),
                    "tca_sec": float(conv.tca_sec),
                    "dca_m": float(conv.dca_m),
                    "is_closing": bool(is_closing),
                    "distance_delta_m": float(distance_delta),
                    "accel_1": float(k1.acceleration_mag),
                    "accel_2": float(k2.acceleration_mag),
                    "speed_1_kmh": float(k1.speed_kmh),
                    "speed_2_kmh": float(k2.speed_kmh),
                    "ground_pos_1": [float(k1.ground_pos[0]), float(k1.ground_pos[1])],
                    "ground_pos_2": [float(k2.ground_pos[0]), float(k2.ground_pos[1])],
                    "velocity_1": [float(k1.velocity_vector[0]), float(k1.velocity_vector[1])],
                    "velocity_2": [float(k2.velocity_vector[0]), float(k2.velocity_vector[1])],
                    "fp_category": fp_cat,
                })

        # Cleanup disappeared pairs
        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
                self.persistence_counts.pop(pk, None)

        # Cleanup stale clusters
        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates


def cluster_episodes(raw_candidates: List[Dict], max_gap_frames: int = 25) -> List[Dict]:
    """Cluster raw candidate triggers into coherent physical incident episodes."""
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
                "fp_category": rep.get("fp_category", "other"),
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
            "fp_category": rep.get("fp_category", "other"),
        })

    return episodes


def evaluate_video_candidates(
    raw_candidates: List[Dict],
    gt_frame: int,
    fps: float,
    tolerance_frames: int = 35,
    max_gap_frames: int = 25,
) -> Tuple[bool, List[Dict], Optional[Dict], Optional[int], Optional[float], int]:
    """Evaluates candidates for a single video against ground truth."""
    episodes = cluster_episodes(raw_candidates, max_gap_frames=max_gap_frames)

    if not episodes:
        return False, [], None, None, None, 0

    matched_episode = None
    fp_episodes = []
    raw_fp_count = 0

    for ep in episodes:
        diff = ep["trigger_frame"] - gt_frame
        if abs(diff) <= tolerance_frames and matched_episode is None:
            matched_episode = ep
        else:
            fp_episodes.append(ep)
            raw_fp_count += ep["raw_count"]

    if matched_episode is not None:
        is_tp = True
        lead_frames = gt_frame - matched_episode["trigger_frame"]
        lead_sec = lead_frames / fps if fps > 0 else 0.0
        return is_tp, fp_episodes, matched_episode, lead_frames, lead_sec, raw_fp_count
    else:
        is_tp = False
        return is_tp, episodes, None, None, None, len(raw_candidates)


def load_or_cache_videos(
    split_type: str,
    split: str,
    max_videos: int,
    cfg,
) -> List[Dict]:
    """Load cached video tracking/kinematics data or generate and cache it."""
    cache_file = f"cache_videos_{split_type}_{max_videos}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading {split_type} videos from local cache: {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Cache not found. Ingesting and caching {max_videos} videos from '{split_type}'...")
    gt_loader = GroundTruthLoader(cfg.evaluation.metadata_path)
    records = gt_loader.filter_split(split=split, split_type=split_type)

    detector = YOLO26Detector(cfg.detection.model_path, conf_threshold=cfg.detection.confidence_threshold, device=cfg.detection.device)
    tracker = ByteTrackTracker()

    video_data = []
    t0 = time.time()
    for idx, gt_rec in enumerate(records[:max_videos], 1):
        v_name = os.path.basename(gt_rec.video_path)
        v_path = os.path.join(cfg.dataset.path, v_name)
        if not os.path.exists(v_path):
            continue

        print(f"  [{idx}/{max_videos}] Ingesting {gt_rec.video_id} (GT={gt_rec.accident_frame})...")
        ingestor = DatasetVideoIngestor(v_path)
        h = PlanarHomography()
        geom = GeometryEngine(h)
        tracker.reset()

        frame_records = []
        fps = 30.0

        for frame in ingestor.stream_video(v_path):
            fps = frame.fps
            batch = detector.detect(frame)
            tl, tracks = tracker.update(batch)
            kin, conv = geom.process_frame(tl, tracks)
            frame_records.append((frame.frame_idx, frame.pts_ns, dict(tracks), dict(kin), list(conv)))

        video_data.append({
            "video_id": gt_rec.video_id,
            "gt_frame": gt_rec.accident_frame,
            "fps": fps,
            "frames": frame_records,
        })

    print(f"Ingestion completed in {time.time() - t0:.1f}s. Saving cache to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(video_data, f)

    return video_data


def run_kinematic_maturity_comparison(
    split_type: str = "split_in_distribution",
    split: str = "test",
    max_videos: int = 15,
    config_path: str = "road_surveillance/config/default_config.yaml",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    video_data = load_or_cache_videos(split_type, split, max_videos, cfg)

    # The 4 target configurations
    configurations = [
        ("0_baseline", "Baseline (Untouched Multi-Signal)"),
        ("1_stage6", "Stage 6 (Spatial Continuity)"),
        ("2_stage6_maturity", "Stage 6 + Track Maturity (age >= 3f)"),
        ("3_stage6_maturity_validity", "Stage 6 + Maturity + Kinematic Validity (Warm-up & Physical Bounds)"),
    ]

    exp_cfg = CollisionConfig(
        max_ground_distance_m=4.0,
        max_perspective_distance_m=12.0,
        min_iou_overlap=0.15,
        min_persistence_frames=2,
        temporal_cluster_gap_frames=25,
    )

    detectors = {
        "1_stage6": KinematicMaturityDetector("stage6", config=exp_cfg),
        "2_stage6_maturity": KinematicMaturityDetector("stage6_maturity", config=exp_cfg),
        "3_stage6_maturity_validity": KinematicMaturityDetector("stage6_maturity_validity", config=exp_cfg, max_physical_decel_ms2=60.0),
    }

    baseline_engine = CollisionCandidateDetector(
        distance_decay_threshold_m=cfg.events.collision.distance_decay_threshold_m,
        tca_threshold_sec=cfg.events.collision.tca_threshold_sec,
        deceleration_threshold_ms2=cfg.events.collision.deceleration_threshold_ms2,
        post_proximity_distance_m=cfg.events.collision.post_proximity_distance_m,
        iou_overlap_threshold=cfg.events.collision.iou_overlap_threshold,
    )

    results = {}
    telemetry_records = []
    fp_breakdowns = {cid: {} for cid, _ in configurations}

    print(f"\nEvaluating {len(configurations)} configurations on '{split_type}' ({len(video_data)} videos)...")

    for cid, clabel in configurations:
        print(f"  Running: {clabel}...")
        tp_count = 0
        fn_count = 0
        fp_episodes_total = 0
        raw_fp_total = 0
        lead_times_frames = []
        lead_times_sec = []
        abs_temp_errors = []

        for v in video_data:
            gt_f = v["gt_frame"]
            fps = v["fps"]
            vid_id = v["video_id"]

            if cid == "0_baseline":
                baseline_engine.prev_distances.clear()
                cands = []
                for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                    raw_c = baseline_engine.evaluate(f_idx, pts_ns, vid_id, tr, kin, conv, LaneManager())
                    for rc in raw_c:
                        t1, t2 = (rc.track_ids[0], rc.track_ids[1]) if len(rc.track_ids) >= 2 else (0, 0)
                        k1 = kin.get(t1)
                        k2 = kin.get(t2)
                        tr1 = tr.get(t1)
                        tr2 = tr.get(t2)
                        conv_match = next((cn for cn in conv if (cn.track_id_1 == t1 and cn.track_id_2 == t2) or (cn.track_id_1 == t2 and cn.track_id_2 == t1)), None)
                        if tr1 and tr2 and k1 and k2 and conv_match:
                            fp_cat = classify_false_positive(tr1, tr2, k1, k2, conv_match, rc.signals.get("ground_distance_m", 0.0), rc.signals.get("bbox_iou", 0.0))
                            s1, s2 = float(k1.speed_kmh), float(k2.speed_kmh)
                            gp1 = [float(k1.ground_pos[0]), float(k1.ground_pos[1])]
                            gp2 = [float(k2.ground_pos[0]), float(k2.ground_pos[1])]
                            v1 = [float(k1.velocity_vector[0]), float(k1.velocity_vector[1])]
                            v2 = [float(k2.velocity_vector[0]), float(k2.velocity_vector[1])]
                        else:
                            fp_cat = "other"
                            s1, s2 = 0.0, 0.0
                            gp1, gp2 = [0.0, 0.0], [0.0, 0.0]
                            v1, v2 = [0.0, 0.0], [0.0, 0.0]

                        cands.append({
                            "frame_idx": rc.frame_idx,
                            "pts_ns": rc.timestamp_ns,
                            "track_ids": rc.track_ids,
                            "dist": rc.signals.get("ground_distance_m", 0.0),
                            "iou": rc.signals.get("bbox_iou", 0.0),
                            "is_converging": rc.signals.get("trajectory_convergence", False),
                            "tca_sec": rc.signals.get("tca_sec", 0.0),
                            "dca_m": rc.signals.get("dca_m", 0.0),
                            "is_closing": rc.signals.get("decreasing_distance", False),
                            "distance_delta_m": rc.signals.get("distance_delta_m", 0.0),
                            "accel_1": rc.signals.get("accel_1_ms2", 0.0),
                            "accel_2": rc.signals.get("accel_2_ms2", 0.0),
                            "speed_1_kmh": s1,
                            "speed_2_kmh": s2,
                            "ground_pos_1": gp1,
                            "ground_pos_2": gp2,
                            "velocity_1": v1,
                            "velocity_2": v2,
                            "fp_category": fp_cat,
                        })
            else:
                det = detectors[cid]
                det.reset()
                cands = []
                for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                    c = det.evaluate_frame(f_idx, pts_ns, tr, kin, conv)
                    cands.extend(c)

            is_tp, fp_eps, matched_tp, lead_f, lead_s, r_fp = evaluate_video_candidates(
                cands, gt_f, fps, tolerance_frames=35, max_gap_frames=25
            )

            if is_tp:
                tp_count += 1
                lead_times_frames.append(lead_f)
                lead_times_sec.append(lead_s)
                abs_temp_errors.append(abs(lead_f))

                rep = matched_tp.get("representative", {})
                telemetry_records.append({
                    "video_id": vid_id,
                    "split_type": split_type,
                    "mode": cid,
                    "status": "TP",
                    "gt_frame": gt_f,
                    "trigger_frame": matched_tp["trigger_frame"],
                    "lead_time_frames": lead_f,
                    "lead_time_sec": round(lead_s, 2),
                    "track_ids": rep.get("track_ids", []),
                    "ground_distance_m": rep.get("dist", 0.0),
                    "bbox_iou": rep.get("iou", 0.0),
                    "accel_1": rep.get("accel_1", 0.0),
                    "accel_2": rep.get("accel_2", 0.0),
                })
            else:
                fn_count += 1

            fp_episodes_total += len(fp_eps)
            raw_fp_total += r_fp

            for ep in fp_eps:
                cat = ep.get("fp_category", "other")
                fp_breakdowns[cid][cat] = fp_breakdowns[cid].get(cat, 0) + 1
                rep = ep.get("representative", {})
                telemetry_records.append({
                    "video_id": vid_id,
                    "split_type": split_type,
                    "mode": cid,
                    "status": "FP",
                    "gt_frame": gt_f,
                    "trigger_frame": ep["trigger_frame"],
                    "fp_category": cat,
                    "track_ids": rep.get("track_ids", []),
                    "ground_distance_m": rep.get("dist", 0.0),
                    "bbox_iou": rep.get("iou", 0.0),
                    "accel_1": rep.get("accel_1", 0.0),
                    "accel_2": rep.get("accel_2", 0.0),
                })

        total_vids = len(video_data)
        prec = tp_count / (tp_count + fp_episodes_total) if (tp_count + fp_episodes_total) > 0 else 0.0
        rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        raw_prec = tp_count / (tp_count + raw_fp_total) if (tp_count + raw_fp_total) > 0 else 0.0
        raw_f1 = (2 * raw_prec * rec) / (raw_prec + rec) if (raw_prec + rec) > 0 else 0.0

        med_lead_f = float(np.median(lead_times_frames)) if lead_times_frames else 0.0
        med_lead_s = float(np.median(lead_times_sec)) if lead_times_sec else 0.0
        med_abs_err = float(np.median(abs_temp_errors)) if abs_temp_errors else 0.0

        results[cid] = {
            "label": clabel,
            "tp": tp_count,
            "fp_episodes": fp_episodes_total,
            "raw_fp": raw_fp_total,
            "fn": fn_count,
            "precision_ep": prec,
            "recall": rec,
            "f1_ep": f1,
            "fp_per_video_ep": fp_episodes_total / total_vids if total_vids > 0 else 0.0,
            "raw_precision": raw_prec,
            "raw_fp_per_video": raw_fp_total / total_vids if total_vids > 0 else 0.0,
            "med_lead_frames": med_lead_f,
            "med_lead_sec": med_lead_s,
            "med_abs_error": med_abs_err,
            "fp_breakdown": fp_breakdowns[cid],
        }

    return {
        "split_type": split_type,
        "video_count": len(video_data),
        "results": results,
        "telemetry": telemetry_records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=15)
    args = parser.parse_args()

    print("=" * 75)
    print("KINEMATIC MATURITY & VALIDITY ABLATION BENCHMARK (BOTH SPLITS)")
    print("=" * 75)

    res_in_dist = run_kinematic_maturity_comparison(
        split_type="split_in_distribution",
        max_videos=args.max_videos,
    )

    res_geo_aware = run_kinematic_maturity_comparison(
        split_type="split_geo_aware",
        max_videos=args.max_videos,
    )

    # Save telemetry
    with open("telemetry_kinematic_maturity_in_dist.json", "w") as f:
        json.dump(res_in_dist["telemetry"], f, indent=2)
    with open("telemetry_kinematic_maturity_geo_aware.json", "w") as f:
        json.dump(res_geo_aware["telemetry"], f, indent=2)

    # Build Markdown Report
    lines = [
        "# Kinematic Maturity Gate & Physical Validity Benchmark Report",
        f"**Evaluated Videos**: 15 per split | **Ground Truth Window**: $\\pm 35$ frames (~1.17s) | **Episode Clustering**: $\\Delta f \\le 25$ frames",
        "",
        "## 1. Episode-Level Deduplicated Performance Comparison",
        "### Split: `split_in_distribution`",
        "| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cid, r in res_in_dist["results"].items():
        lines.append(
            f"| **{r['label']}** | {r['tp']} | {r['fp_episodes']} | {r['fn']} | "
            f"**{r['precision_ep']:.3f}** | **{r['recall']:.3f}** | **{r['f1_ep']:.3f}** | "
            f"{r['fp_per_video_ep']:.2f} | +{r['med_lead_sec']:.2f}s ({r['med_lead_frames']:+.0f}f) | {r['med_abs_error']:.0f} frames |"
        )

    lines.extend([
        "",
        "### Split: `split_geo_aware`",
        "| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for cid, r in res_geo_aware["results"].items():
        lines.append(
            f"| **{r['label']}** | {r['tp']} | {r['fp_episodes']} | {r['fn']} | "
            f"**{r['precision_ep']:.3f}** | **{r['recall']:.3f}** | **{r['f1_ep']:.3f}** | "
            f"{r['fp_per_video_ep']:.2f} | +{r['med_lead_sec']:.2f}s ({r['med_lead_frames']:+.0f}f) | {r['med_abs_error']:.0f} frames |"
        )

    lines.extend([
        "",
        "## 2. Impact of Deduplication: Episode Clustering vs. Raw Frame Triggers",
        "| Split | Configuration | Raw FP | Raw FP / Vid | Raw Prec | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for s_name, s_res in [("In-Dist", res_in_dist), ("Geo-Aware", res_geo_aware)]:
        for cid, r in s_res["results"].items():
            gain = (r['precision_ep'] / r['raw_precision']) if r['raw_precision'] > 0 else 1.0
            lines.append(
                f"| **{s_name}** | **{r['label']}** | {r['raw_fp']} | {r['raw_fp_per_video']:.1f} | {r['raw_precision']:.3f} | "
                f"{r['fp_episodes']} | {r['fp_per_video_ep']:.2f} | **{r['precision_ep']:.3f}** | **{gain:.1f}x** |"
            )

    lines.extend([
        "",
        "## 3. False Positive Failure Mode Taxonomy Breakdown (Episodes)",
        "| Split | Configuration | Normal Following | Lane Merge | Crossing Traffic | Stopped Traffic | Tracker Artifact | Calibration Artifact | Other | Total FP Ep |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    categories = ["normal_following", "lane_merge", "crossing_traffic", "stopped_traffic", "tracker_artifact", "calibration_artifact", "other"]

    for s_name, s_res in [("In-Dist", res_in_dist), ("Geo-Aware", res_geo_aware)]:
        for cid, r in s_res["results"].items():
            b = r["fp_breakdown"]
            row = f"| **{s_name}** | **{r['label']}** | "
            for cat in categories:
                row += f"{b.get(cat, 0)} | "
            row += f"**{r['fp_episodes']}** |"
            lines.append(row)

    md_out = "\n".join(lines)
    print("\n" + md_out + "\n")

    with open("docs/kinematic_maturity_ablation_report.md", "w", encoding="utf-8") as f:
        f.write(md_out)
    print("Report saved to: docs/kinematic_maturity_ablation_report.md")


if __name__ == "__main__":
    main()
