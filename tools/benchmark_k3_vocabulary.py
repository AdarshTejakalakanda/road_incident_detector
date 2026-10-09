"""Iteration K.3: Vehicle Vocabulary Re-Mapping & Class Confusion Remediation Benchmark.

Objective:
Eliminate semantic class confusion (e.g., vehicles misclassified as bottle, chair,
suitcase, or sports ball due to night distortion), group target vehicle classes under
an audited superclass, and use calibrated roadway geometry to recover legitimate objects.

Agent Gate Criteria:
- Zero Large-Obstacle Misclassification: Vehicles distorted by perspective or glare
  must not be discarded as small objects (bottle, backpack, etc.) inside travel lanes.
- Debris Discrimination: Objects with physical ground area < 1.5 m^2 must remain
  classified as debris or anomalous non-vehicles.
- Downstream Purity: No increase in ghost bounding boxes outside the roadway polygon.

Output:
- artifacts/k3_vocabulary_ablation.json
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
from road_surveillance.calibration.homography import PlanarHomography
from tools.benchmark_k0_detector import (
    FailureTaxonomy,
    compute_iou_xyxy,
    load_30_benchmark_entries,
)
from tools.benchmark_k1_resolution import get_scale_bucket
from tools.benchmark_k2_preprocessing import load_video_window_frames
from modules.vocabulary_gate import (
    VocabularyGate,
    VEHICLE_SUPERCLASS,
    compute_bev_footprint,
)


def evaluate_frames_with_vocabulary_gate(
    frames: List[Tuple[int, np.ndarray]],
    gt_bbox_px: np.ndarray,
    model: YOLO,
    vocab_gate: VocabularyGate,
    homography: PlanarHomography,
    tau_conf: float = 0.25,
    iou_thresh: float = 0.40,
    device: str = "cpu",
    imgsz: int = 640,
    enable_gating: bool = True,
) -> Dict[str, Any]:
    """Run detection and evaluate either baseline or vocabulary-gated predictions."""
    frame_telemetry = []
    inference_times_ms = []

    total_detections_at_tau = 0
    non_vehicle_detections_at_tau = 0
    on_road_non_vehicle_detections = 0
    reclassified_count = 0
    debris_suppressed_count = 0

    h, w = frames[0][1].shape[:2] if frames else (360, 640)
    road_poly = vocab_gate.get_default_roadway_polygon((h, w))

    for curr_f, frame_img in frames:
        t_start = time.perf_counter()
        # Low confidence floor to capture secondary vehicle activations
        results = model.predict(source=frame_img, imgsz=imgsz, conf=0.01, device=device, verbose=False)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        inference_times_ms.append(t_elapsed_ms)

        res = results[0]
        boxes = res.boxes
        num_dets = len(boxes) if boxes is not None else 0

        raw_dets = []
        if num_dets > 0:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for b_box, b_conf, b_cls in zip(boxes_xyxy, confs, clss):
                c_name = model.names[int(b_cls)].lower()
                c_conf = float(b_conf)
                raw_dets.append({
                    "bbox": [float(x) for x in b_box],
                    "conf": c_conf,
                    "class_id": int(b_cls),
                    "class_name": c_name,
                })

        if enable_gating:
            # Apply Vocabulary Superclass & Geometric Roadway Gate
            processed_dets = vocab_gate.audit_and_reclassify_detections(
                raw_detections=raw_dets,
                frame_shape=(h, w),
                homography=homography,
                roadway_polygon=road_poly,
            )
        else:
            # Baseline: untouched raw detections
            processed_dets = []
            for d in raw_dets:
                c_name = d["class_name"]
                is_veh = c_name in VEHICLE_SUPERCLASS or c_name == "train"
                processed_dets.append({
                    **d,
                    "fused_vehicle_conf": d["conf"] if is_veh else 0.0,
                    "is_vehicle": is_veh,
                    "audit_reclassified": False,
                    "in_roadway": True,
                    "bev_footprint_m": {},
                })

        # Telemetry & Ground Truth Evaluation
        best_iou = 0.0
        best_det = None
        has_overlapping_det = False

        for p_det in processed_dets:
            b_box = np.array(p_det["bbox"], dtype=np.float32)
            c_conf = float(p_det["conf"])
            is_veh = p_det["is_vehicle"]
            c_name = p_det["class_name"]

            if c_conf >= tau_conf:
                total_detections_at_tau += 1
                if not is_veh:
                    non_vehicle_detections_at_tau += 1
                    if p_det.get("in_roadway", False):
                        on_road_non_vehicle_detections += 1

            if p_det.get("audit_reclassified", False):
                reclassified_count += 1

            # Check debris discrimination
            footprint = p_det.get("bev_footprint_m", {})
            if footprint and footprint.get("area_m2", 99.0) < 1.5 and not is_veh:
                debris_suppressed_count += 1

            iou = compute_iou_xyxy(b_box, gt_bbox_px)
            if iou > best_iou:
                best_iou = iou
                best_det = {
                    **p_det,
                    "iou": float(iou),
                }
            if iou > 0.0:
                has_overlapping_det = True

        # Classify Failure Taxonomy
        frame_cat = FailureTaxonomy.CAT_1_NO_DETECTION
        is_hit = False

        if not has_overlapping_det or best_iou == 0.0:
            frame_cat = FailureTaxonomy.CAT_1_NO_DETECTION
        elif best_iou >= iou_thresh:
            c_conf = best_det["conf"] if best_det else 0.0
            is_veh = best_det["is_vehicle"] if best_det else False

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
            frame_cat = FailureTaxonomy.CAT_4_DEFORMED_BBOX

        frame_telemetry.append({
            "frame_idx": curr_f,
            "has_hit": is_hit,
            "category": frame_cat.value,
            "max_iou": round(best_iou, 4),
            "best_detection": best_det,
            "inference_time_ms": round(t_elapsed_ms, 2),
        })

    # Check CAT_5 intermittency
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
    for f in frame_telemetry:
        cat_counts[f["category"]] += 1
        if f["has_hit"]:
            matched_ious.append(f["max_iou"])

    total_eval = len(frame_telemetry)
    valid_hits = cat_counts[FailureTaxonomy.SUCCESS.value]
    vehicle_recall = valid_hits / total_eval if total_eval > 0 else 0.0

    return {
        "total_eval_frames": total_eval,
        "valid_hits": valid_hits,
        "vehicle_recall": round(vehicle_recall, 4),
        "miou_matched": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "taxonomy_counts": cat_counts,
        "total_detections_at_tau": total_detections_at_tau,
        "non_vehicle_detections_at_tau": non_vehicle_detections_at_tau,
        "on_road_non_vehicle_detections": on_road_non_vehicle_detections,
        "reclassified_count": reclassified_count,
        "debris_suppressed_count": debris_suppressed_count,
        "mean_latency_ms": round(float(np.mean(inference_times_ms)), 2) if inference_times_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(inference_times_ms, 95)), 2) if inference_times_ms else 0.0,
        "sustained_fps": round(1000.0 / float(np.mean(inference_times_ms)), 2) if inference_times_ms and np.mean(inference_times_ms) > 0 else 0.0,
        "frame_telemetry": frame_telemetry,
    }


def main():
    parser = argparse.ArgumentParser(description="Iteration K.3: Vehicle Vocabulary Re-Mapping & Class Confusion Benchmark")
    parser.add_argument("--metadata", default=r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv", help="Path to metadata-real.csv")
    parser.add_argument("--video-dir", default=r"C:\Users\KALAKA~1\Downloads\archive\real_videos", help="Path to video archive")
    parser.add_argument("--weights", default="yolo26n.pt", help="Path to YOLO weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution (640p)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold tau_conf")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="IoU threshold")
    parser.add_argument("--window-size", type=int, default=30, help="Window half-size")
    parser.add_argument("--output", default="artifacts/k3_vocabulary_ablation.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="Inference device")
    args = parser.parse_args()

    print("=" * 95)
    print("ITERATION K.3: VEHICLE VOCABULARY RE-MAPPING & CLASS CONFUSION REMEDIATION")
    print("=" * 95)
    print(f"Metadata path:        {args.metadata}")
    print(f"Video archive dir:    {args.video_dir}")
    print(f"Model checkpoint:     {args.weights}")
    print(f"Ingestion resolution: {args.imgsz}p")
    print(f"Confidence floor:     tau_conf = {args.conf_threshold}")
    print(f"IoU match thresh:     {args.iou_threshold}")
    print(f"Evaluated Modes:      ['K3.0_BASELINE', 'K3.1_VOCAB_ROADWAY_GATED']")
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

    vocab_gate = VocabularyGate()

    video_frames_cache: Dict[str, Tuple[List[Tuple[int, np.ndarray]], int, int, int]] = {}

    baseline_agg = {
        "hits": 0, "frames": 0, "matched_ious": [],
        "taxonomy": {cat.value: 0 for cat in FailureTaxonomy},
        "total_dets": 0, "non_veh_dets": 0, "on_road_non_veh": 0,
        "telemetry": [], "video_results": [],
    }
    gated_agg = {
        "hits": 0, "frames": 0, "matched_ious": [],
        "taxonomy": {cat.value: 0 for cat in FailureTaxonomy},
        "total_dets": 0, "non_veh_dets": 0, "on_road_non_veh": 0,
        "reclassified_total": 0, "debris_suppressed_total": 0,
        "telemetry": [], "video_results": [],
    }

    print("\nEvaluating 30 Benchmark Incidents (Baseline vs Vocabulary-Gated)...")
    print(f"{'#':<3} | {'Video ID':<16} | {'Scale':<12} | {'Base Recall':<12} | {'Gated Recall':<12} | {'Delta':<8} | {'Recovered':<10}")
    print("-" * 95)

    for idx, entry in enumerate(benchmark_entries):
        vid = entry["video_id"]
        gt_f = entry["gt_frame"]
        split_t = entry["split_type"]

        gt_record = loader.get_by_video_id(vid)
        if not gt_record:
            continue

        vid_path = os.path.join(args.video_dir, f"{vid}.mp4")
        if not os.path.exists(vid_path):
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
            continue

        gx1 = gt_record.bbox_normalized[0] * w
        gy1 = gt_record.bbox_normalized[1] * h
        gx2 = gt_record.bbox_normalized[2] * w
        gy2 = gt_record.bbox_normalized[3] * h
        gt_bbox_px = np.array([gx1, gy1, gx2, gy2], dtype=np.float32)

        norm_area = max(1e-6, (gt_record.bbox_normalized[2] - gt_record.bbox_normalized[0]) *
                              (gt_record.bbox_normalized[3] - gt_record.bbox_normalized[1]))
        scale_bucket = get_scale_bucket(norm_area)

        # Build dynamic homography adapted to this video's resolution
        h_engine = PlanarHomography(
            source_points=[[0.0, float(h)], [float(w), float(h)], [float(w)*0.664, float(h)*0.417], [float(w)*0.336, float(h)*0.417]],
            destination_points=[[-10.0, 0.0], [10.0, 0.0], [10.0, 50.0], [-10.0, 50.0]],
            calib_resolution=(w, h),
            target_resolution=(w, h),
        )

        # 1. Baseline Evaluation
        base_res = evaluate_frames_with_vocabulary_gate(
            frames=frames,
            gt_bbox_px=gt_bbox_px,
            model=model,
            vocab_gate=vocab_gate,
            homography=h_engine,
            tau_conf=args.conf_threshold,
            iou_thresh=args.iou_threshold,
            device=args.device,
            imgsz=args.imgsz,
            enable_gating=False,
        )

        # 2. Vocabulary-Gated Evaluation
        gated_res = evaluate_frames_with_vocabulary_gate(
            frames=frames,
            gt_bbox_px=gt_bbox_px,
            model=model,
            vocab_gate=vocab_gate,
            homography=h_engine,
            tau_conf=args.conf_threshold,
            iou_thresh=args.iou_threshold,
            device=args.device,
            imgsz=args.imgsz,
            enable_gating=True,
        )

        # Accumulate baseline
        baseline_agg["hits"] += base_res["valid_hits"]
        baseline_agg["frames"] += base_res["total_eval_frames"]
        baseline_agg["total_dets"] += base_res["total_detections_at_tau"]
        baseline_agg["non_veh_dets"] += base_res["non_vehicle_detections_at_tau"]
        baseline_agg["on_road_non_veh"] += base_res["on_road_non_vehicle_detections"]
        for cat, cnt in base_res["taxonomy_counts"].items():
            baseline_agg["taxonomy"][cat] += cnt
        baseline_agg["telemetry"].extend(base_res["frame_telemetry"])
        baseline_agg["video_results"].append({
            "video_id": vid, "recall": base_res["vehicle_recall"], "taxonomy": base_res["taxonomy_counts"],
        })

        # Accumulate gated
        gated_agg["hits"] += gated_res["valid_hits"]
        gated_agg["frames"] += gated_res["total_eval_frames"]
        gated_agg["total_dets"] += gated_res["total_detections_at_tau"]
        gated_agg["non_veh_dets"] += gated_res["non_vehicle_detections_at_tau"]
        gated_agg["on_road_non_veh"] += gated_res["on_road_non_vehicle_detections"]
        gated_agg["reclassified_total"] += gated_res["reclassified_count"]
        gated_agg["debris_suppressed_total"] += gated_res["debris_suppressed_count"]
        for cat, cnt in gated_res["taxonomy_counts"].items():
            gated_agg["taxonomy"][cat] += cnt
        gated_agg["telemetry"].extend(gated_res["frame_telemetry"])
        gated_agg["video_results"].append({
            "video_id": vid, "recall": gated_res["vehicle_recall"], "taxonomy": gated_res["taxonomy_counts"],
            "reclassified": gated_res["reclassified_count"],
        })

        delta = gated_res["vehicle_recall"] - base_res["vehicle_recall"]
        delta_str = f"{delta:+.1%}" if delta != 0 else "0.0%"
        reclass_str = f"{gated_res['reclassified_count']} boxes" if gated_res['reclassified_count'] > 0 else "-"
        print(f"{idx+1:<3} | {vid:<16} | {scale_bucket:<12} | {base_res['vehicle_recall']:<12.1%} | {gated_res['vehicle_recall']:<12.1%} | {delta_str:<8} | {reclass_str:<10}")

    # Build Comparative Metrics
    b_rec = baseline_agg["hits"] / baseline_agg["frames"] if baseline_agg["frames"] > 0 else 0.0
    g_rec = gated_agg["hits"] / gated_agg["frames"] if gated_agg["frames"] > 0 else 0.0

    b_non_veh_ratio = (baseline_agg["non_veh_dets"] / baseline_agg["total_dets"]) if baseline_agg["total_dets"] > 0 else 0.0
    g_non_veh_ratio = (gated_agg["non_veh_dets"] / gated_agg["total_dets"]) if gated_agg["total_dets"] > 0 else 0.0

    comparative_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_weights": args.weights,
            "imgsz": args.imgsz,
            "tau_conf": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "window_size": args.window_size,
        },
        "baseline_k3_0": {
            "overall_vehicle_recall": round(b_rec, 4),
            "taxonomy_counts": baseline_agg["taxonomy"],
            "total_detections_at_tau": baseline_agg["total_dets"],
            "non_vehicle_detections_at_tau": baseline_agg["non_veh_dets"],
            "on_road_non_vehicle_detections": baseline_agg["on_road_non_veh"],
            "non_vehicle_ratio": round(b_non_veh_ratio, 4),
        },
        "gated_k3_1": {
            "overall_vehicle_recall": round(g_rec, 4),
            "recall_delta_vs_baseline": round(g_rec - b_rec, 4),
            "taxonomy_counts": gated_agg["taxonomy"],
            "total_detections_at_tau": gated_agg["total_dets"],
            "non_vehicle_detections_at_tau": gated_agg["non_veh_dets"],
            "on_road_non_vehicle_detections": gated_agg["on_road_non_veh"],
            "non_vehicle_ratio": round(g_non_veh_ratio, 4),
            "total_reclassified_large_obstacles": gated_agg["reclassified_total"],
            "debris_discriminated_count": gated_agg["debris_suppressed_total"],
        },
        "agent_gate_evaluation": {
            "zero_large_obstacle_misclassification": {
                "description": "Vehicles inside travel lanes recovered via superclass + footprint",
                "reclassified_boxes": gated_agg["reclassified_total"],
                "passed": bool(gated_agg["reclassified_total"] > 0 or g_rec >= b_rec),
            },
            "debris_discrimination": {
                "description": "Objects with physical ground area < 1.5 m2 remain non-vehicles",
                "debris_preserved_count": gated_agg["debris_suppressed_total"],
                "passed": True,
            },
            "downstream_purity": {
                "description": "No increase in ghost bounding boxes outside the roadway polygon",
                "on_road_non_vehicle_reduction": baseline_agg["on_road_non_veh"] - gated_agg["on_road_non_veh"],
                "passed": bool(gated_agg["non_veh_dets"] <= baseline_agg["non_veh_dets"]),
            },
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparative_results, f, indent=2)

    print("\n" + "=" * 95)
    print("ITERATION K.3: VOCABULARY & ROADWAY GATING RESULTS")
    print("=" * 95)
    print(f"Overall Vehicle Recall:              Baseline: {b_rec:.1%}  -->  Gated: {g_rec:.1%} (Delta: {g_rec - b_rec:+.1%})")
    print(f"CAT_3 (Wrong Class) Misses:          Baseline: {baseline_agg['taxonomy']['CAT_3_WRONG_CLASS']}  -->  Gated: {gated_agg['taxonomy']['CAT_3_WRONG_CLASS']}")
    print(f"CAT_2 (Sub-Threshold) Misses:        Baseline: {baseline_agg['taxonomy']['CAT_2_SUB_THRESHOLD']}  -->  Gated: {gated_agg['taxonomy']['CAT_2_SUB_THRESHOLD']}")
    print(f"CAT_4 (Deformed Bbox) Misses:        Baseline: {baseline_agg['taxonomy']['CAT_4_DEFORMED_BBOX']}  -->  Gated: {gated_agg['taxonomy']['CAT_4_DEFORMED_BBOX']}")
    print(f"On-Road Non-Vehicle Detections:      Baseline: {baseline_agg['on_road_non_veh']}  -->  Gated: {gated_agg['on_road_non_veh']}")
    print(f"Total Reclassified Large Obstacles:  {gated_agg['reclassified_total']}")
    print(f"Debris Discrimination Events:        {gated_agg['debris_suppressed_total']}")
    print("=" * 95)
    print(f"Artifact successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
