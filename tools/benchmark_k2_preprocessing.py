"""Iteration K.2: Illumination-Gated Preprocessing Pipeline Benchmark.

Objective:
Evaluate whether targeted low-light enhancement (Gamma adjustment, CLAHE on luminance,
and adaptive routing) pulls CAT_2 detections above threshold and sharpens vehicle edges
to convert CAT_4 into valid IoU matches, without inducing false positives or adding
> 2 ms processing overhead.

Experimental Configurations (Evaluated at 640p):
- K2.0 (Frozen Baseline): Raw frame (no preprocessing).
- K2.1 (Global Gamma Correction): I_out = 255 * (I_in / 255)^0.65
- K2.2 (Selective CLAHE on Luminance): LAB L-channel CLAHE (clipLimit=2.0, tileGridSize=(8,8))
- K2.3 (Illumination-Gated Adaptive Preprocessing):
    - DAY_NORMAL -> Pass-through (Raw)
    - LOW_LIGHT -> Gamma (gamma = 0.65)
    - NIGHT_HIGH_GLARE -> CLAHE (L-channel, clipLimit=1.5) + Mild Gamma (gamma = 0.80)

Agent Gate Criteria:
- Overall Vehicle Recall: Absolute increase >= +3.0% over K2.0 (Baseline: ~13.1%)
- CAT_2 Transition: Relative reduction of CAT_2_SUB_THRESHOLD by >= 15%
- CAT_4 Recovery: Measure how many CAT_4 frames convert to valid matches (IoU >= 0.40)
- Ghost False Positives: Non-vehicle detections inside roadway < 5%
- Latency Penalty: Preprocessing overhead <= 2.5 ms (sustained >= 21 FPS)

Output:
- artifacts/k2_preprocessing_ablation.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from ultralytics import YOLO

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from road_surveillance.evaluation import GroundTruthLoader
from tools.benchmark_k0_detector import (
    FailureTaxonomy,
    VEHICLE_CLASSES,
    compute_iou_xyxy,
    load_30_benchmark_entries,
)
from tools.benchmark_k1_resolution import get_scale_bucket
from modules.lighting_gate import LightingGate, LightingCondition


CONFIGURATIONS = {
    "K2.0_RAW": "raw",
    "K2.1_GAMMA": "gamma",
    "K2.2_CLAHE": "clahe",
    "K2.3_ADAPTIVE": "adaptive",
}


def load_video_window_frames(
    video_path: str,
    gt_frame: int,
    window_half_size: int = 30,
) -> Tuple[List[Tuple[int, np.ndarray]], int, int, int]:
    """Read video frames in the window [t_GT - 30, t_GT + 30] into memory once."""
    if not os.path.exists(video_path):
        return [], 0, 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(1, gt_frame - window_half_size)
    end_frame = min(total_frames, gt_frame + window_half_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    frames = []
    curr_f = start_frame - 1
    while curr_f < end_frame:
        ret, frame_img = cap.read()
        if not ret or frame_img is None:
            break
        curr_f += 1
        frames.append((curr_f, frame_img))

    cap.release()
    return frames, total_frames, w, h


def evaluate_frames_with_preprocessor(
    frames: List[Tuple[int, np.ndarray]],
    gt_bbox_px: np.ndarray,
    model: YOLO,
    preprocessor: LightingGate,
    tau_conf: float = 0.25,
    iou_thresh: float = 0.40,
    device: str = "cpu",
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Run preprocessing, detector inference, and failure taxonomy on a sequence of frames."""
    frame_telemetry = []
    preprocess_times_ms = []
    inference_times_ms = []
    total_times_ms = []

    total_detections_at_tau = 0
    non_vehicle_detections_at_tau = 0

    for curr_f, raw_frame in frames:
        # Preprocessing step
        t_pre_start = time.perf_counter()
        proc_frame, pre_tel = preprocessor.process(raw_frame)
        t_pre_ms = (time.perf_counter() - t_pre_start) * 1000.0
        preprocess_times_ms.append(t_pre_ms)

        # Inference step
        t_inf_start = time.perf_counter()
        results = model.predict(source=proc_frame, imgsz=imgsz, conf=0.01, device=device, verbose=False)
        t_inf_ms = (time.perf_counter() - t_inf_start) * 1000.0
        inference_times_ms.append(t_inf_ms)
        total_times_ms.append(t_pre_ms + t_inf_ms)

        res = results[0]
        boxes = res.boxes
        num_dets = len(boxes) if boxes is not None else 0

        best_iou = 0.0
        best_det = None
        has_overlapping_det = False

        if num_dets > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for b_box, b_conf, b_cls in zip(boxes_xyxy, confs, clss):
                c_name = model.names[int(b_cls)].lower()
                c_conf = float(b_conf)

                # Track ghost detections (non-vehicle with conf >= tau_conf)
                if c_conf >= tau_conf:
                    total_detections_at_tau += 1
                    if c_name not in VEHICLE_CLASSES and c_name != "train":
                        non_vehicle_detections_at_tau += 1

                iou = compute_iou_xyxy(b_box, gt_bbox_px)
                if iou > best_iou:
                    best_iou = iou
                    best_det = {
                        "bbox": [float(x) for x in b_box],
                        "conf": c_conf,
                        "class_id": int(b_cls),
                        "class_name": c_name,
                        "iou": float(iou),
                    }
                if iou > 0.0:
                    has_overlapping_det = True

        # Frame level taxonomy classification
        frame_cat = FailureTaxonomy.CAT_1_NO_DETECTION
        is_hit = False

        if not has_overlapping_det or best_iou == 0.0:
            frame_cat = FailureTaxonomy.CAT_1_NO_DETECTION
        elif best_iou >= iou_thresh:
            c_name = best_det["class_name"].lower() if best_det else ""
            c_conf = best_det["conf"] if best_det else 0.0
            is_veh = c_name in VEHICLE_CLASSES or c_name == "train"

            if not is_veh and c_conf >= tau_conf:
                frame_cat = FailureTaxonomy.CAT_3_WRONG_CLASS
            elif c_conf < tau_conf:
                frame_cat = FailureTaxonomy.CAT_2_SUB_THRESHOLD
            elif is_veh and c_conf >= tau_conf:
                frame_cat = FailureTaxonomy.SUCCESS
                is_hit = True
            else:
                frame_cat = FailureTaxonomy.CAT_2_SUB_THRESHOLD
        else:
            # 0.0 < best_iou < iou_thresh
            c_name = best_det["class_name"].lower() if best_det else ""
            is_veh = c_name in VEHICLE_CLASSES or c_name == "train"
            frame_cat = FailureTaxonomy.CAT_4_DEFORMED_BBOX

        frame_telemetry.append({
            "frame_idx": curr_f,
            "has_hit": is_hit,
            "category": frame_cat.value,
            "max_iou": round(best_iou, 4),
            "best_detection": best_det,
            "preprocessor_telemetry": pre_tel,
            "preprocess_time_ms": round(t_pre_ms, 3),
            "inference_time_ms": round(t_inf_ms, 3),
            "total_frame_time_ms": round(t_pre_ms + t_inf_ms, 3),
        })

    # Apply CAT_5_INTERMITTENT check (<= 3 hits over 10 consecutive frames)
    n_frames = len(frame_telemetry)
    hit_flags = [f["has_hit"] for f in frame_telemetry]

    for i in range(n_frames):
        w_start = max(0, i - 4)
        w_end = min(n_frames, w_start + 10)
        if w_end - w_start < 10 and w_start > 0:
            w_start = max(0, w_end - 10)
        window_hits = sum(hit_flags[w_start:w_end])

        if frame_telemetry[i]["category"] == FailureTaxonomy.SUCCESS.value:
            if 0 < window_hits <= 3:
                frame_telemetry[i]["category"] = FailureTaxonomy.CAT_5_INTERMITTENT.value

    cat_counts = {cat.value: 0 for cat in FailureTaxonomy}
    matched_ious = []
    overlapping_ious = []

    for f in frame_telemetry:
        cat_counts[f["category"]] += 1
        m_iou = f["max_iou"]
        if m_iou > 0.0:
            overlapping_ious.append(m_iou)
        if f["has_hit"]:
            matched_ious.append(m_iou)

    total_eval = len(frame_telemetry)
    valid_hits = cat_counts[FailureTaxonomy.SUCCESS.value]
    vehicle_recall = valid_hits / total_eval if total_eval > 0 else 0.0

    return {
        "total_eval_frames": total_eval,
        "valid_hits": valid_hits,
        "vehicle_recall": round(vehicle_recall, 4),
        "miou_matched": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "miou_overlapping": round(float(np.mean(overlapping_ious)), 4) if overlapping_ious else 0.0,
        "taxonomy_counts": cat_counts,
        "mean_preprocess_ms": round(float(np.mean(preprocess_times_ms)), 2) if preprocess_times_ms else 0.0,
        "p95_preprocess_ms": round(float(np.percentile(preprocess_times_ms, 95)), 2) if preprocess_times_ms else 0.0,
        "mean_inference_ms": round(float(np.mean(inference_times_ms)), 2) if inference_times_ms else 0.0,
        "p95_inference_ms": round(float(np.percentile(inference_times_ms, 95)), 2) if inference_times_ms else 0.0,
        "mean_total_ms": round(float(np.mean(total_times_ms)), 2) if total_times_ms else 0.0,
        "p95_total_ms": round(float(np.percentile(total_times_ms, 95)), 2) if total_times_ms else 0.0,
        "sustained_fps": round(1000.0 / float(np.mean(total_times_ms)), 2) if total_times_ms and np.mean(total_times_ms) > 0 else 0.0,
        "total_detections_at_tau": total_detections_at_tau,
        "non_vehicle_detections_at_tau": non_vehicle_detections_at_tau,
        "frame_telemetry": frame_telemetry,
    }


def compute_transition_matrix(
    baseline_telemetry: List[Dict[str, Any]],
    enhanced_telemetry: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute frame-level taxonomy transitions from Baseline (K2.0) to Enhanced."""
    categories = [cat.value for cat in FailureTaxonomy]
    transitions = {f"{c_from}->{c_to}": 0 for c_from in categories for c_to in categories}

    cat2_conversions_to_success = 0
    cat4_conversions_to_success = 0
    cat1_conversions_to_success = 0
    regressions_from_success = 0

    total_frames = min(len(baseline_telemetry), len(enhanced_telemetry))
    for i in range(total_frames):
        b_cat = baseline_telemetry[i]["category"]
        e_cat = enhanced_telemetry[i]["category"]
        transitions[f"{b_cat}->{e_cat}"] += 1

        if b_cat == FailureTaxonomy.CAT_2_SUB_THRESHOLD.value and e_cat == FailureTaxonomy.SUCCESS.value:
            cat2_conversions_to_success += 1
        if b_cat == FailureTaxonomy.CAT_4_DEFORMED_BBOX.value and e_cat == FailureTaxonomy.SUCCESS.value:
            cat4_conversions_to_success += 1
        if b_cat == FailureTaxonomy.CAT_1_NO_DETECTION.value and e_cat == FailureTaxonomy.SUCCESS.value:
            cat1_conversions_to_success += 1
        if b_cat == FailureTaxonomy.SUCCESS.value and e_cat != FailureTaxonomy.SUCCESS.value:
            regressions_from_success += 1

    # Keep only non-zero transitions for clean serialization
    active_transitions = {k: v for k, v in transitions.items() if v > 0}

    return {
        "cat2_to_success": cat2_conversions_to_success,
        "cat4_to_success": cat4_conversions_to_success,
        "cat1_to_success": cat1_conversions_to_success,
        "regressions_from_success": regressions_from_success,
        "detailed_transitions": active_transitions,
    }


def main():
    parser = argparse.ArgumentParser(description="Iteration K.2: Illumination-Gated Preprocessing Pipeline Benchmark")
    parser.add_argument("--metadata", default=r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv", help="Path to metadata-real.csv")
    parser.add_argument("--video-dir", default=r"C:\Users\KALAKA~1\Downloads\archive\real_videos", help="Path to video archive")
    parser.add_argument("--weights", default="yolo26n.pt", help="Path to YOLO weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Ingestion resolution (frozen at 640p)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Baseline confidence threshold tau_conf")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="IoU threshold for valid hit / failure taxonomy")
    parser.add_argument("--window-size", type=int, default=30, help="Half-window size around GT frame")
    parser.add_argument("--output", default="artifacts/k2_preprocessing_ablation.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device")
    args = parser.parse_args()

    print("=" * 95)
    print("ITERATION K.2: ILLUMINATION-GATED PREPROCESSING PIPELINE BENCHMARK")
    print("=" * 95)
    print(f"Metadata path:        {args.metadata}")
    print(f"Video archive dir:    {args.video_dir}")
    print(f"Model checkpoint:     {args.weights}")
    print(f"Ingestion resolution: {args.imgsz}p (FROZEN)")
    print(f"Confidence floor:     tau_conf = {args.conf_threshold}")
    print(f"IoU match thresh:     {args.iou_threshold}")
    print(f"Evaluated Modes:      {list(CONFIGURATIONS.keys())}")
    print(f"Device:               {args.device}")
    print(f"Output artifact:      {args.output}")
    print("-" * 95)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("Loading Ground Truth and 30 Benchmark Video Entries...")
    loader = GroundTruthLoader(args.metadata)
    benchmark_entries = load_30_benchmark_entries(args.metadata)
    print(f"Loaded {len(benchmark_entries)} benchmark video entries.")

    print(f"Initializing YOLO26 from '{args.weights}'...")
    model = YOLO(args.weights)

    # Initialize preprocessor engines
    preprocessors = {
        cfg_name: LightingGate(mode=mode_str)
        for cfg_name, mode_str in CONFIGURATIONS.items()
    }

    # Data structures for results
    results_by_config: Dict[str, Dict[str, Any]] = {
        cfg_name: {
            "config_name": cfg_name,
            "mode": mode_str,
            "total_frames_evaluated": 0,
            "total_valid_hits": 0,
            "scale_buckets": {
                "micro": {"hits": 0, "frames": 0},
                "tiny": {"hits": 0, "frames": 0},
                "small": {"hits": 0, "frames": 0},
                "medium_large": {"hits": 0, "frames": 0},
            },
            "taxonomy_counts": {cat.value: 0 for cat in FailureTaxonomy},
            "matched_ious": [],
            "overlapping_ious": [],
            "preprocess_times": [],
            "inference_times": [],
            "total_times": [],
            "total_detections_at_tau": 0,
            "non_vehicle_detections_at_tau": 0,
            "all_frame_telemetry": [],
            "video_results": [],
        }
        for cfg_name, mode_str in CONFIGURATIONS.items()
    }

    # Cache for loaded frames and video metrics
    video_frames_cache: Dict[str, Tuple[List[Tuple[int, np.ndarray]], int, int, int]] = {}

    print("\nStarting Unified Preprocessing Evaluation across 30 Incidents...")
    print(f"{'#':<3} | {'Video ID':<16} | {'Scale':<12} | {'K2.0 Recall':<12} | {'K2.1 Recall':<12} | {'K2.2 Recall':<12} | {'K2.3 Recall':<12}")
    print("-" * 95)

    for idx, entry in enumerate(benchmark_entries):
        vid = entry["video_id"]
        gt_f = entry["gt_frame"]
        split_t = entry["split_type"]

        gt_record = loader.get_by_video_id(vid)
        if not gt_record:
            print(f"Warning: No GT record for {vid}")
            continue

        vid_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(vid_path):
            print(f"Warning: Video file missing: {vid_path}")
            continue

        cache_key = f"{vid}_{gt_f}"
        if cache_key in video_frames_cache:
            frames, total_frames, w, h = video_frames_cache[cache_key]
        else:
            frames, total_frames, w, h = load_video_window_frames(
                video_path=vid_path,
                gt_frame=gt_f,
                window_half_size=args.window_size,
            )
            video_frames_cache[cache_key] = (frames, total_frames, w, h)

        if not frames:
            print(f"Error loading frames for {vid}")
            continue

        # Ground truth bounding box
        gx1 = gt_record.bbox_normalized[0] * w
        gy1 = gt_record.bbox_normalized[1] * h
        gx2 = gt_record.bbox_normalized[2] * w
        gy2 = gt_record.bbox_normalized[3] * h
        gt_bbox_px = np.array([gx1, gy1, gx2, gy2], dtype=np.float32)

        norm_area = max(1e-6, (gt_record.bbox_normalized[2] - gt_record.bbox_normalized[0]) *
                              (gt_record.bbox_normalized[3] - gt_record.bbox_normalized[1]))
        scale_bucket = get_scale_bucket(norm_area)

        # Run each configuration on this video segment
        video_recalls = {}
        for cfg_name, preprocessor in preprocessors.items():
            eval_out = evaluate_frames_with_preprocessor(
                frames=frames,
                gt_bbox_px=gt_bbox_px,
                model=model,
                preprocessor=preprocessor,
                tau_conf=args.conf_threshold,
                iou_thresh=args.iou_threshold,
                device=args.device,
                imgsz=args.imgsz,
            )

            cfg_data = results_by_config[cfg_name]
            cfg_data["total_frames_evaluated"] += eval_out["total_eval_frames"]
            cfg_data["total_valid_hits"] += eval_out["valid_hits"]
            cfg_data["scale_buckets"][scale_bucket]["hits"] += eval_out["valid_hits"]
            cfg_data["scale_buckets"][scale_bucket]["frames"] += eval_out["total_eval_frames"]

            for cat, cnt in eval_out["taxonomy_counts"].items():
                cfg_data["taxonomy_counts"][cat] += cnt

            for ft in eval_out["frame_telemetry"]:
                cfg_data["preprocess_times"].append(ft["preprocess_time_ms"])
                cfg_data["inference_times"].append(ft["inference_time_ms"])
                cfg_data["total_times"].append(ft["total_frame_time_ms"])
                cfg_data["all_frame_telemetry"].append(ft)
                if ft["has_hit"]:
                    cfg_data["matched_ious"].append(ft["max_iou"])
                if ft["max_iou"] > 0.0:
                    cfg_data["overlapping_ious"].append(ft["max_iou"])

            cfg_data["total_detections_at_tau"] += eval_out["total_detections_at_tau"]
            cfg_data["non_vehicle_detections_at_tau"] += eval_out["non_vehicle_detections_at_tau"]

            video_recalls[cfg_name] = eval_out["vehicle_recall"]
            cfg_data["video_results"].append({
                "index": idx + 1,
                "video_id": vid,
                "split_type": split_t,
                "scale_bucket": scale_bucket,
                "norm_area_pct": round(norm_area * 100.0, 4),
                "gt_frame": gt_f,
                "total_eval_frames": eval_out["total_eval_frames"],
                "valid_hits": eval_out["valid_hits"],
                "vehicle_recall": eval_out["vehicle_recall"],
                "miou_matched": eval_out["miou_matched"],
                "taxonomy_counts": eval_out["taxonomy_counts"],
            })

        print(f"{idx+1:<3} | {vid:<16} | {scale_bucket:<12} | "
              f"{video_recalls['K2.0_RAW']:<12.1%} | "
              f"{video_recalls['K2.1_GAMMA']:<12.1%} | "
              f"{video_recalls['K2.2_CLAHE']:<12.1%} | "
              f"{video_recalls['K2.3_ADAPTIVE']:<12.1%}")

    # Build Comparative Matrix and Transitions
    baseline_telemetry = results_by_config["K2.0_RAW"]["all_frame_telemetry"]
    baseline_recall = results_by_config["K2.0_RAW"]["total_valid_hits"] / results_by_config["K2.0_RAW"]["total_frames_evaluated"]
    baseline_cat2 = results_by_config["K2.0_RAW"]["taxonomy_counts"][FailureTaxonomy.CAT_2_SUB_THRESHOLD.value]

    comparative_matrix = {}
    transitions_by_config = {}

    for cfg_name, cfg_data in results_by_config.items():
        n_eval = cfg_data["total_frames_evaluated"]
        n_hits = cfg_data["total_valid_hits"]
        overall_rec = n_hits / n_eval if n_eval > 0 else 0.0

        sb = cfg_data["scale_buckets"]
        micro_tiny_hits = sb["micro"]["hits"] + sb["tiny"]["hits"]
        micro_tiny_frames = sb["micro"]["frames"] + sb["tiny"]["frames"]
        micro_tiny_rec = micro_tiny_hits / micro_tiny_frames if micro_tiny_frames > 0 else 0.0

        small_hits = sb["small"]["hits"]
        small_frames = sb["small"]["frames"]
        small_rec = small_hits / small_frames if small_frames > 0 else 0.0

        med_large_hits = sb["medium_large"]["hits"]
        med_large_frames = sb["medium_large"]["frames"]
        med_large_rec = med_large_hits / med_large_frames if med_large_frames > 0 else 0.0

        miou_m = float(np.mean(cfg_data["matched_ious"])) if cfg_data["matched_ious"] else 0.0
        p95_pre = float(np.percentile(cfg_data["preprocess_times"], 95)) if cfg_data["preprocess_times"] else 0.0
        mean_pre = float(np.mean(cfg_data["preprocess_times"])) if cfg_data["preprocess_times"] else 0.0
        p95_total = float(np.percentile(cfg_data["total_times"], 95)) if cfg_data["total_times"] else 0.0
        mean_total = float(np.mean(cfg_data["total_times"])) if cfg_data["total_times"] else 0.0
        sustained_fps = 1000.0 / mean_total if mean_total > 0 else 0.0

        tot_dets = cfg_data["total_detections_at_tau"]
        non_veh_dets = cfg_data["non_vehicle_detections_at_tau"]
        ghost_fp_ratio = (non_veh_dets / tot_dets) if tot_dets > 0 else 0.0

        # Transitions relative to K2.0
        trans = compute_transition_matrix(baseline_telemetry, cfg_data["all_frame_telemetry"])
        transitions_by_config[cfg_name] = trans

        cat2_cnt = cfg_data["taxonomy_counts"][FailureTaxonomy.CAT_2_SUB_THRESHOLD.value]
        cat2_reduction_rel = ((baseline_cat2 - cat2_cnt) / baseline_cat2) if baseline_cat2 > 0 else 0.0

        comparative_matrix[cfg_name] = {
            "overall_vehicle_recall": round(overall_rec, 4),
            "recall_delta_vs_baseline": round(overall_rec - baseline_recall, 4),
            "micro_tiny_recall": round(micro_tiny_rec, 4),
            "small_bucket_recall": round(small_rec, 4),
            "medium_large_recall": round(med_large_rec, 4),
            "miou_matched": round(miou_m, 4),
            "cat1_no_detection": cfg_data["taxonomy_counts"][FailureTaxonomy.CAT_1_NO_DETECTION.value],
            "cat2_sub_threshold": cat2_cnt,
            "cat2_reduction_relative": round(cat2_reduction_rel, 4),
            "cat2_converted_to_success": trans["cat2_to_success"],
            "cat4_deformed_bbox": cfg_data["taxonomy_counts"][FailureTaxonomy.CAT_4_DEFORMED_BBOX.value],
            "cat4_converted_to_success": trans["cat4_to_success"],
            "cat5_intermittent": cfg_data["taxonomy_counts"][FailureTaxonomy.CAT_5_INTERMITTENT.value],
            "ghost_fp_ratio": round(ghost_fp_ratio, 4),
            "ghost_fp_count": non_veh_dets,
            "total_detections_conf_ge_025": tot_dets,
            "mean_preprocessing_latency_ms": round(mean_pre, 2),
            "p95_preprocessing_latency_ms": round(p95_pre, 2),
            "mean_total_latency_ms": round(mean_total, 2),
            "p95_total_latency_ms": round(p95_total, 2),
            "sustained_fps": round(sustained_fps, 2),
        }

    # Evaluate Agent Gate Criteria for K2.3 (Adaptive)
    k23_metrics = comparative_matrix["K2.3_ADAPTIVE"]
    recall_gate = bool(k23_metrics["recall_delta_vs_baseline"] >= 0.030)
    cat2_gate = bool(k23_metrics["cat2_reduction_relative"] >= 0.150)
    ghost_gate = bool(k23_metrics["ghost_fp_ratio"] < 0.050)
    latency_gate = bool(k23_metrics["mean_preprocessing_latency_ms"] <= 2.5 and k23_metrics["sustained_fps"] >= 21.0)

    # Determine recommended configuration
    # Check best trade-off among K2.1, K2.2, K2.3
    best_config = "K2.0_RAW"
    best_score = baseline_recall
    for c_name in ["K2.1_GAMMA", "K2.2_CLAHE", "K2.3_ADAPTIVE"]:
        c_m = comparative_matrix[c_name]
        if c_m["overall_vehicle_recall"] > best_score and c_m["sustained_fps"] >= 18.0 and c_m["ghost_fp_ratio"] < 0.05:
            best_score = c_m["overall_vehicle_recall"]
            best_config = c_name

    final_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_weights": args.weights,
            "imgsz": args.imgsz,
            "tau_conf": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "window_size": args.window_size,
            "device": args.device,
            "tested_configurations": list(CONFIGURATIONS.keys()),
        },
        "agent_gate_evaluation": {
            "overall_vehicle_recall_gate": {
                "threshold": ">= +3.0% abs gain",
                "k23_delta": round(k23_metrics["recall_delta_vs_baseline"] * 100, 2),
                "passed": recall_gate,
            },
            "cat2_sub_threshold_reduction_gate": {
                "threshold": ">= 15.0% relative reduction",
                "k23_reduction": round(k23_metrics["cat2_reduction_relative"] * 100, 2),
                "passed": cat2_gate,
            },
            "cat4_recovery": {
                "k23_cat4_to_success_frames": k23_metrics["cat4_converted_to_success"],
            },
            "ghost_false_positives_gate": {
                "threshold": "< 5.0% non-vehicle detections",
                "k23_ratio": round(k23_metrics["ghost_fp_ratio"] * 100, 2),
                "passed": ghost_gate,
            },
            "latency_penalty_gate": {
                "threshold": "<= 2.5 ms preprocessing & >= 21.0 FPS sustained",
                "k23_preprocessing_latency_ms": k23_metrics["mean_preprocessing_latency_ms"],
                "k23_sustained_fps": k23_metrics["sustained_fps"],
                "passed": latency_gate,
            },
            "recommended_configuration": best_config,
        },
        "comparative_matrix": comparative_matrix,
        "transitions": transitions_by_config,
        "video_level_results": {
            cfg_name: cfg_data["video_results"]
            for cfg_name, cfg_data in results_by_config.items()
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 105)
    print("ITERATION K.2: COMPARATIVE PREPROCESSING ABLATION MATRIX")
    print("=" * 105)
    print(f"{'Metric / Parameter':<32} | {'K2.0 (Raw)':<16} | {'K2.1 (Gamma)':<16} | {'K2.2 (CLAHE)':<16} | {'K2.3 (Adaptive)':<16}")
    print("-" * 105)

    m = comparative_matrix
    print(f"{'Overall Vehicle Recall':<32} | {m['K2.0_RAW']['overall_vehicle_recall']:<16.1%} | {m['K2.1_GAMMA']['overall_vehicle_recall']:<16.1%} | {m['K2.2_CLAHE']['overall_vehicle_recall']:<16.1%} | {m['K2.3_ADAPTIVE']['overall_vehicle_recall']:<16.1%}")
    print(f"{'Recall Delta vs K2.0':<32} | {'0.0%':<16} | {m['K2.1_GAMMA']['recall_delta_vs_baseline']:<+16.1%} | {m['K2.2_CLAHE']['recall_delta_vs_baseline']:<+16.1%} | {m['K2.3_ADAPTIVE']['recall_delta_vs_baseline']:<+16.1%}")
    print(f"{'Small-Bucket Recall (0.1-0.25%)':<32} | {m['K2.0_RAW']['small_bucket_recall']:<16.1%} | {m['K2.1_GAMMA']['small_bucket_recall']:<16.1%} | {m['K2.2_CLAHE']['small_bucket_recall']:<16.1%} | {m['K2.3_ADAPTIVE']['small_bucket_recall']:<16.1%}")
    print(f"{'Medium/Large Recall (>=0.25%)':<32} | {m['K2.0_RAW']['medium_large_recall']:<16.1%} | {m['K2.1_GAMMA']['medium_large_recall']:<16.1%} | {m['K2.2_CLAHE']['medium_large_recall']:<16.1%} | {m['K2.3_ADAPTIVE']['medium_large_recall']:<16.1%}")
    print(f"{'Mean IoU on Matched Boxes':<32} | {m['K2.0_RAW']['miou_matched']:<16.4f} | {m['K2.1_GAMMA']['miou_matched']:<16.4f} | {m['K2.2_CLAHE']['miou_matched']:<16.4f} | {m['K2.3_ADAPTIVE']['miou_matched']:<16.4f}")
    print("-" * 105)
    print(f"{'CAT_1 (No Detection) Frames':<32} | {m['K2.0_RAW']['cat1_no_detection']:<16} | {m['K2.1_GAMMA']['cat1_no_detection']:<16} | {m['K2.2_CLAHE']['cat1_no_detection']:<16} | {m['K2.3_ADAPTIVE']['cat1_no_detection']:<16}")
    print(f"{'CAT_2 (Sub-Threshold) Frames':<32} | {m['K2.0_RAW']['cat2_sub_threshold']:<16} | {m['K2.1_GAMMA']['cat2_sub_threshold']:<16} | {m['K2.2_CLAHE']['cat2_sub_threshold']:<16} | {m['K2.3_ADAPTIVE']['cat2_sub_threshold']:<16}")
    print(f"{'CAT_2 Relative Reduction':<32} | {'0.0%':<16} | {m['K2.1_GAMMA']['cat2_reduction_relative']:<16.1%} | {m['K2.2_CLAHE']['cat2_reduction_relative']:<16.1%} | {m['K2.3_ADAPTIVE']['cat2_reduction_relative']:<16.1%}")
    print(f"{'CAT_2 -> SUCCESS Recovered':<32} | {'-':<16} | {m['K2.1_GAMMA']['cat2_converted_to_success']:<16} | {m['K2.2_CLAHE']['cat2_converted_to_success']:<16} | {m['K2.3_ADAPTIVE']['cat2_converted_to_success']:<16}")
    print(f"{'CAT_4 (Deformed Bbox) Frames':<32} | {m['K2.0_RAW']['cat4_deformed_bbox']:<16} | {m['K2.1_GAMMA']['cat4_deformed_bbox']:<16} | {m['K2.2_CLAHE']['cat4_deformed_bbox']:<16} | {m['K2.3_ADAPTIVE']['cat4_deformed_bbox']:<16}")
    print(f"{'CAT_4 -> SUCCESS Recovered':<32} | {'-':<16} | {m['K2.1_GAMMA']['cat4_converted_to_success']:<16} | {m['K2.2_CLAHE']['cat4_converted_to_success']:<16} | {m['K2.3_ADAPTIVE']['cat4_converted_to_success']:<16}")
    print(f"{'CAT_5 (Intermittent) Frames':<32} | {m['K2.0_RAW']['cat5_intermittent']:<16} | {m['K2.1_GAMMA']['cat5_intermittent']:<16} | {m['K2.2_CLAHE']['cat5_intermittent']:<16} | {m['K2.3_ADAPTIVE']['cat5_intermittent']:<16}")
    print("-" * 105)
    print(f"{'Ghost False Positive Ratio':<32} | {m['K2.0_RAW']['ghost_fp_ratio']:<16.2%} | {m['K2.1_GAMMA']['ghost_fp_ratio']:<16.2%} | {m['K2.2_CLAHE']['ghost_fp_ratio']:<16.2%} | {m['K2.3_ADAPTIVE']['ghost_fp_ratio']:<16.2%}")
    print(f"{'Mean Preprocessing Latency (ms)':<32} | {m['K2.0_RAW']['mean_preprocessing_latency_ms']:<16.2f} | {m['K2.1_GAMMA']['mean_preprocessing_latency_ms']:<16.2f} | {m['K2.2_CLAHE']['mean_preprocessing_latency_ms']:<16.2f} | {m['K2.3_ADAPTIVE']['mean_preprocessing_latency_ms']:<16.2f}")
    print(f"{'P95 Preprocessing Latency (ms)':<32} | {m['K2.0_RAW']['p95_preprocessing_latency_ms']:<16.2f} | {m['K2.1_GAMMA']['p95_preprocessing_latency_ms']:<16.2f} | {m['K2.2_CLAHE']['p95_preprocessing_latency_ms']:<16.2f} | {m['K2.3_ADAPTIVE']['p95_preprocessing_latency_ms']:<16.2f}")
    print(f"{'Sustained Inference Speed (FPS)':<32} | {m['K2.0_RAW']['sustained_fps']:<16.1f} | {m['K2.1_GAMMA']['sustained_fps']:<16.1f} | {m['K2.2_CLAHE']['sustained_fps']:<16.1f} | {m['K2.3_ADAPTIVE']['sustained_fps']:<16.1f}")
    print("=" * 105)
    print(f"Artifact successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
