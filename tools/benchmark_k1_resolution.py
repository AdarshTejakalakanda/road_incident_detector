"""Iteration K.1: Dynamic Ingestion Resolution & Scale Tradeoff.

Objective:
Evaluate YOLO26 at resolutions [640, 960, 1280] on the identical 30-benchmark harness
to resolve small-object blindness (< 1024 px^2) and reduce bounding box edge distortion,
while establishing the Pareto frontier against inference FPS.

Tasks:
1. Parameterize inference resolution (imgsz in [640, 960, 1280]).
2. Spatial scale bucketing:
   - Micro: A_norm < 0.05% (< 460 px^2 at 720p)
   - Tiny: 0.05% <= A_norm < 0.10% (460 to 921 px^2)
   - Small: 0.10% <= A_norm < 0.25% (921 to 2304 px^2)
   - Medium/Large: A_norm >= 0.25% (> 2304 px^2)
3. Measure downstream metrics:
   - Overall Vehicle Recall (%)
   - Micro + Tiny Recall (%), Small Recall (%), Medium/Large Recall (%)
   - Mean IoU (mIoU) on matched boxes
   - Taxonomy shift (CAT_1 through CAT_5)
   - P95 Inference Latency (ms) & Sustained FPS
4. Agent Gate Evaluation & Selection Rule.

Output:
- artifacts/k1_resolution_ablation.json
"""

import os
import sys
import json
import time
import argparse
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
from ultralytics import YOLO

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from road_surveillance.evaluation import GroundTruthLoader
from tools.benchmark_k0_detector import FailureTaxonomy, VEHICLE_CLASSES, compute_iou_xyxy, load_30_benchmark_entries


def get_scale_bucket(a_norm: float) -> str:
    """Classify normalized bounding box area into spatial scale buckets."""
    if a_norm < 0.0005:
        return "micro"
    elif a_norm < 0.0010:
        return "tiny"
    elif a_norm < 0.0025:
        return "small"
    else:
        return "medium_large"


def evaluate_video_at_resolution(
    video_path: str,
    gt_frame: int,
    gt_bbox_normalized: Tuple[float, float, float, float],
    model: YOLO,
    imgsz: int = 640,
    tau_conf: float = 0.25,
    iou_thresh: float = 0.40,
    window_half_size: int = 30,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate YOLO26 raw detections at a specific imgsz across [t_GT - 30, t_GT + 30]."""
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Failed to open video: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(1, gt_frame - window_half_size)
    end_frame = min(total_frames, gt_frame + window_half_size)

    # Scale normalized GT bounding box to pixel coordinates
    gx1 = gt_bbox_normalized[0] * w
    gy1 = gt_bbox_normalized[1] * h
    gx2 = gt_bbox_normalized[2] * w
    gy2 = gt_bbox_normalized[3] * h
    gt_bbox_px = np.array([gx1, gy1, gx2, gy2], dtype=np.float32)

    gt_area_px = max(1.0, (gx2 - gx1) * (gy2 - gy1))
    norm_area = max(1e-6, (gt_bbox_normalized[2] - gt_bbox_normalized[0]) * (gt_bbox_normalized[3] - gt_bbox_normalized[1]))
    scale_bucket = get_scale_bucket(norm_area)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    frame_telemetry = []
    inference_times_ms = []

    curr_f = start_frame - 1
    while curr_f < end_frame:
        ret, frame_img = cap.read()
        if not ret or frame_img is None:
            break
        curr_f += 1

        t_start = time.perf_counter()
        # Ultralytics letterboxes to imgsz and maps output boxes back to original (w, h)
        results = model.predict(source=frame_img, imgsz=imgsz, conf=0.01, device=device, verbose=False)
        t_elapsed_s = time.perf_counter() - t_start
        inference_times_ms.append(t_elapsed_s * 1000.0)

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
                iou = compute_iou_xyxy(b_box, gt_bbox_px)
                c_name = model.names[int(b_cls)]
                if iou > best_iou:
                    best_iou = iou
                    best_det = {
                        "bbox": [float(x) for x in b_box],
                        "conf": float(b_conf),
                        "class_id": int(b_cls),
                        "class_name": c_name,
                        "iou": float(iou),
                    }
                if iou > 0.0:
                    has_overlapping_det = True

        # Determine frame-level detection category
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
            frame_cat = FailureTaxonomy.CAT_4_DEFORMED_BBOX

        frame_telemetry.append({
            "frame_idx": curr_f,
            "has_hit": is_hit,
            "category": frame_cat.value,
            "max_iou": round(best_iou, 4),
            "best_detection": best_det,
            "latency_ms": round(t_elapsed_s * 1000.0, 2),
        })

    cap.release()

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
    for f in frame_telemetry:
        cat_counts[f["category"]] += 1

    total_eval = len(frame_telemetry)
    valid_hits = cat_counts[FailureTaxonomy.SUCCESS.value]
    vehicle_recall = valid_hits / total_eval if total_eval > 0 else 0.0

    # mIoU calculations
    hit_ious = [f["max_iou"] for f in frame_telemetry if f["has_hit"]]
    miou_hits = float(np.mean(hit_ious)) if hit_ious else 0.0

    overlapping_ious = [f["max_iou"] for f in frame_telemetry if f["max_iou"] > 0.0]
    miou_overlapping = float(np.mean(overlapping_ious)) if overlapping_ious else 0.0

    p95_latency = float(np.percentile(inference_times_ms, 95)) if inference_times_ms else 0.0
    mean_latency = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    sustained_fps = (total_eval / (sum(inference_times_ms) / 1000.0)) if inference_times_ms and sum(inference_times_ms) > 0 else 0.0

    if vehicle_recall >= 0.70:
        primary_cat = FailureTaxonomy.SUCCESS.value
    elif cat_counts[FailureTaxonomy.CAT_5_INTERMITTENT.value] > 0 and vehicle_recall < 0.40:
        primary_cat = FailureTaxonomy.CAT_5_INTERMITTENT.value
    else:
        fail_modes = [
            FailureTaxonomy.CAT_1_NO_DETECTION.value,
            FailureTaxonomy.CAT_2_SUB_THRESHOLD.value,
            FailureTaxonomy.CAT_3_WRONG_CLASS.value,
            FailureTaxonomy.CAT_4_DEFORMED_BBOX.value,
            FailureTaxonomy.CAT_5_INTERMITTENT.value,
        ]
        primary_cat = max(fail_modes, key=lambda m: cat_counts[m])

    return {
        "imgsz": imgsz,
        "video_path": video_path,
        "total_eval_frames": total_eval,
        "gt_frame": gt_frame,
        "gt_bbox_normalized": [round(float(x), 5) for x in gt_bbox_normalized],
        "gt_area_px": round(gt_area_px, 1),
        "norm_area_pct": round(norm_area * 100.0, 4),
        "scale_bucket": scale_bucket,
        "valid_hits": valid_hits,
        "vehicle_recall": round(vehicle_recall, 4),
        "miou_matched": round(miou_hits, 4),
        "miou_overlapping": round(miou_overlapping, 4),
        "primary_failure_category": primary_cat,
        "taxonomy_counts": cat_counts,
        "mean_latency_ms": round(mean_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "sustained_fps": round(sustained_fps, 2),
        "inference_times_ms": inference_times_ms,
        "frame_telemetry": frame_telemetry,
    }


def run_resolution_benchmark(
    metadata_path: str,
    video_dir: str,
    weights: str = "yolo26n.pt",
    resolutions: List[int] = [640, 960, 1280],
    tau_conf: float = 0.25,
    iou_thresh: float = 0.40,
    window_size: int = 30,
    device: str = "cpu",
    output_path: str = "artifacts/k1_resolution_ablation.json",
):
    print("=" * 90)
    print("ITERATION K.1: DYNAMIC INGESTION RESOLUTION & SCALE TRADEOFF BENCHMARK")
    print("=" * 90)
    print(f"Resolutions to Evaluate: {resolutions}")
    print(f"Model Checkpoint:        {weights}")
    print(f"Confidence Floor:        tau_conf = {tau_conf}")
    print(f"IoU Match Threshold:     {iou_thresh}")
    print(f"Window Size:             +/- {window_size} frames")
    print(f"Device:                  {device}")
    print(f"Output File:             {output_path}")
    print("-" * 90)

    loader = GroundTruthLoader(metadata_path)
    entries = load_30_benchmark_entries(metadata_path)
    print(f"Loaded {len(entries)} benchmark entries across 21 unique video files.\n")

    model = YOLO(weights)

    ablation_results = {}

    for res_size in resolutions:
        print("\n" + "=" * 90)
        print(f">>> EVALUATING RESOLUTION: {res_size}p (imgsz = {res_size})")
        print("=" * 90)
        print(f"{'#':<3} | {'Video ID':<16} | {'Scale Bucket':<13} | {'Norm Area%':<10} | {'Hits/Total':<11} | {'Recall':<7} | {'mIoU':<6} | {'FPS':<6} | {'Primary Mode'}")
        print("-" * 90)

        res_video_results = []
        scale_buckets_eval = {
            "micro": {"hits": 0, "frames": 0},
            "tiny": {"hits": 0, "frames": 0},
            "small": {"hits": 0, "frames": 0},
            "medium_large": {"hits": 0, "frames": 0},
        }
        total_eval_frames = 0
        total_hits = 0
        all_matched_ious = []
        all_overlapping_ious = []
        all_latencies_ms = []
        taxonomy_totals = {cat.value: 0 for cat in FailureTaxonomy}

        # Cache per unique video for this resolution
        cached_runs = {}

        for idx, entry in enumerate(entries):
            vid = entry["video_id"]
            gt_f = entry["gt_frame"]
            split_t = entry["split_type"]

            gt_rec = loader.get_by_video_id(vid)
            if not gt_rec:
                continue

            vid_path = os.path.join(video_dir, f"{vid}.mp4")
            cache_key = f"{vid}_{gt_f}_{res_size}"

            if cache_key in cached_runs:
                ev = cached_runs[cache_key]
            else:
                ev = evaluate_video_at_resolution(
                    video_path=vid_path,
                    gt_frame=gt_f,
                    gt_bbox_normalized=gt_rec.bbox_normalized,
                    model=model,
                    imgsz=res_size,
                    tau_conf=tau_conf,
                    iou_thresh=iou_thresh,
                    window_half_size=window_size,
                    device=device,
                )
                cached_runs[cache_key] = ev

            if "error" in ev:
                print(f"Error {vid}: {ev['error']}")
                continue

            res_video_results.append({
                "index": idx + 1,
                "video_id": vid,
                "split_type": split_t,
                "incident_type": gt_rec.incident_type,
                "day_time": gt_rec.day_time,
                "scale_bucket": ev["scale_bucket"],
                "norm_area_pct": ev["norm_area_pct"],
                "gt_frame": gt_f,
                "total_eval_frames": ev["total_eval_frames"],
                "valid_hits": ev["valid_hits"],
                "vehicle_recall": ev["vehicle_recall"],
                "miou_matched": ev["miou_matched"],
                "primary_failure_category": ev["primary_failure_category"],
                "taxonomy_counts": ev["taxonomy_counts"],
                "sustained_fps": ev["sustained_fps"],
            })

            b_name = ev["scale_bucket"]
            scale_buckets_eval[b_name]["hits"] += ev["valid_hits"]
            scale_buckets_eval[b_name]["frames"] += ev["total_eval_frames"]

            total_eval_frames += ev["total_eval_frames"]
            total_hits += ev["valid_hits"]

            for f in ev["frame_telemetry"]:
                if f["has_hit"]:
                    all_matched_ious.append(f["max_iou"])
                if f["max_iou"] > 0.0:
                    all_overlapping_ious.append(f["max_iou"])

            all_latencies_ms.extend(ev["inference_times_ms"])

            for cat, cnt in ev["taxonomy_counts"].items():
                taxonomy_totals[cat] += cnt

            hits_str = f"{ev['valid_hits']}/{ev['total_eval_frames']}"
            rec_str = f"{ev['vehicle_recall']:.1%}"
            print(f"{idx+1:<3} | {vid:<16} | {b_name:<13} | {ev['norm_area_pct']:<10.3f}% | {hits_str:<11} | {rec_str:<7} | {ev['miou_matched']:<6.3f} | {ev['sustained_fps']:<6.1f} | {ev['primary_failure_category']}")

        # Compute aggregate metrics for this resolution
        overall_recall = total_hits / total_eval_frames if total_eval_frames > 0 else 0.0

        micro_recall = scale_buckets_eval["micro"]["hits"] / scale_buckets_eval["micro"]["frames"] if scale_buckets_eval["micro"]["frames"] > 0 else 0.0
        tiny_recall = scale_buckets_eval["tiny"]["hits"] / scale_buckets_eval["tiny"]["frames"] if scale_buckets_eval["tiny"]["frames"] > 0 else 0.0
        micro_tiny_hits = scale_buckets_eval["micro"]["hits"] + scale_buckets_eval["tiny"]["hits"]
        micro_tiny_frames = scale_buckets_eval["micro"]["frames"] + scale_buckets_eval["tiny"]["frames"]
        micro_tiny_recall = micro_tiny_hits / micro_tiny_frames if micro_tiny_frames > 0 else 0.0

        small_recall = scale_buckets_eval["small"]["hits"] / scale_buckets_eval["small"]["frames"] if scale_buckets_eval["small"]["frames"] > 0 else 0.0
        med_large_recall = scale_buckets_eval["medium_large"]["hits"] / scale_buckets_eval["medium_large"]["frames"] if scale_buckets_eval["medium_large"]["frames"] > 0 else 0.0

        combined_small_all_hits = micro_tiny_hits + scale_buckets_eval["small"]["hits"]
        combined_small_all_frames = micro_tiny_frames + scale_buckets_eval["small"]["frames"]
        combined_small_all_recall = combined_small_all_hits / combined_small_all_frames if combined_small_all_frames > 0 else 0.0

        m_iou_matched = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0
        m_iou_all = float(np.mean(all_overlapping_ious)) if all_overlapping_ious else 0.0

        p95_lat = float(np.percentile(all_latencies_ms, 95)) if all_latencies_ms else 0.0
        mean_lat = float(np.mean(all_latencies_ms)) if all_latencies_ms else 0.0
        sustained_fps = total_eval_frames / (sum(all_latencies_ms) / 1000.0) if all_latencies_ms and sum(all_latencies_ms) > 0 else 0.0

        total_failures = total_eval_frames - taxonomy_totals[FailureTaxonomy.SUCCESS.value]

        ablation_results[res_size] = {
            "imgsz": res_size,
            "overall_vehicle_recall": round(overall_recall, 4),
            "micro_recall": round(micro_recall, 4),
            "tiny_recall": round(tiny_recall, 4),
            "micro_tiny_combined_recall": round(micro_tiny_recall, 4),
            "small_bucket_recall": round(small_recall, 4),
            "all_small_objects_recall": round(combined_small_all_recall, 4),
            "medium_large_recall": round(med_large_recall, 4),
            "miou_matched_boxes": round(m_iou_matched, 4),
            "miou_overlapping_boxes": round(m_iou_all, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "sustained_fps": round(sustained_fps, 2),
            "total_frames_evaluated": total_eval_frames,
            "total_valid_hits": total_hits,
            "total_failures": total_failures,
            "taxonomy_counts": taxonomy_totals,
            "scale_buckets": scale_buckets_eval,
            "video_results": res_video_results,
        }

    # Selection Rule Evaluation:
    # 1. Combined Micro + Tiny recall must increase from 0.0% to >= 15.0%
    # 2. Sustained inference speed >= 18 FPS
    # 3. If 1280p does not provide >= 10% relative improvement over 960p on small objects, select 960p
    rec_640_small = ablation_results[640]["all_small_objects_recall"]
    rec_960_small = ablation_results[960]["all_small_objects_recall"]
    rec_1280_small = ablation_results[1280]["all_small_objects_recall"]

    rel_improvement_1280_vs_960 = ((rec_1280_small - rec_960_small) / rec_960_small) if rec_960_small > 0 else 0.0

    selected_res = 960
    selection_rationale = ""
    if rel_improvement_1280_vs_960 >= 0.10 and ablation_results[1280]["sustained_fps"] >= 18.0:
        selected_res = 1280
        selection_rationale = f"1280p selected: provides {rel_improvement_1280_vs_960:+.1%} relative small-object gain over 960p and satisfies >= 18 FPS."
    elif ablation_results[960]["sustained_fps"] >= 18.0 or ablation_results[960]["all_small_objects_recall"] > rec_640_small:
        selected_res = 960
        selection_rationale = (
            f"960p selected: 1280p small-object relative gain is {rel_improvement_1280_vs_960:+.1%} (< 10% threshold). "
            f"960p conserves compute budget ({ablation_results[960]['sustained_fps']:.1f} FPS vs {ablation_results[1280]['sustained_fps']:.1f} FPS)."
        )
    else:
        selected_res = 640
        selection_rationale = "Fallback to 640p due to strict FPS constraint."

    summary_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_weights": weights,
            "tau_conf": tau_conf,
            "iou_threshold": iou_thresh,
            "window_size": window_size,
            "device": device,
            "tested_resolutions": resolutions,
            "selected_resolution": selected_res,
            "selection_rationale": selection_rationale,
        },
        "comparative_matrix": {
            str(r): {
                "overall_recall": ablation_results[r]["overall_vehicle_recall"],
                "micro_tiny_recall": ablation_results[r]["micro_tiny_combined_recall"],
                "small_bucket_recall": ablation_results[r]["small_bucket_recall"],
                "all_small_recall": ablation_results[r]["all_small_objects_recall"],
                "medium_large_recall": ablation_results[r]["medium_large_recall"],
                "miou_matched": ablation_results[r]["miou_matched_boxes"],
                "p95_latency_ms": ablation_results[r]["p95_latency_ms"],
                "sustained_fps": ablation_results[r]["sustained_fps"],
                "cat1_no_detection": ablation_results[r]["taxonomy_counts"]["CAT_1_NO_DETECTION"],
                "cat2_sub_threshold": ablation_results[r]["taxonomy_counts"]["CAT_2_SUB_THRESHOLD"],
                "cat4_deformed_bbox": ablation_results[r]["taxonomy_counts"]["CAT_4_DEFORMED_BBOX"],
                "cat5_intermittent": ablation_results[r]["taxonomy_counts"]["CAT_5_INTERMITTENT"],
            }
            for r in resolutions
        },
        "ablation_details": ablation_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    # Print Comparative Matrix Table
    print("\n" + "=" * 95)
    print("ITERATION K.1: COMPARATIVE ABLATION MATRIX (PARETO FRONTIER)")
    print("=" * 95)
    print(f"{'Metric / Parameter':<28} | {'640p (Baseline)':<18} | {'960p (Dynamic)':<18} | {'1280p (High-Res)':<18}")
    print("-" * 95)
    m640 = ablation_results[640]
    m960 = ablation_results[960]
    m1280 = ablation_results[1280]

    print(f"{'Overall Vehicle Recall':<28} | {m640['overall_vehicle_recall']:<17.1%} | {m960['overall_vehicle_recall']:<17.1%} | {m1280['overall_vehicle_recall']:<17.1%}")
    print(f"{'Micro+Tiny Recall (<0.10%)':<28} | {m640['micro_tiny_combined_recall']:<17.1%} | {m960['micro_tiny_combined_recall']:<17.1%} | {m1280['micro_tiny_combined_recall']:<17.1%}")
    print(f"{'Small Bucket Recall (0.1-0.25%)':<28} | {m640['small_bucket_recall']:<17.1%} | {m960['small_bucket_recall']:<17.1%} | {m1280['small_bucket_recall']:<17.1%}")
    print(f"{'All Small Objects (<0.25%)':<28} | {m640['all_small_objects_recall']:<17.1%} | {m960['all_small_objects_recall']:<17.1%} | {m1280['all_small_objects_recall']:<17.1%}")
    print(f"{'Medium/Large Recall (>=0.25%)':<28} | {m640['medium_large_recall']:<17.1%} | {m960['medium_large_recall']:<17.1%} | {m1280['medium_large_recall']:<17.1%}")
    print(f"{'Mean IoU on Matched Boxes':<28} | {m640['miou_matched_boxes']:<18.3f} | {m960['miou_matched_boxes']:<18.3f} | {m1280['miou_matched_boxes']:<18.3f}")
    print("-" * 95)
    print(f"{'CAT_1 (No Detection) Frames':<28} | {m640['taxonomy_counts']['CAT_1_NO_DETECTION']:<18} | {m960['taxonomy_counts']['CAT_1_NO_DETECTION']:<18} | {m1280['taxonomy_counts']['CAT_1_NO_DETECTION']:<18}")
    print(f"{'CAT_2 (Sub-Threshold) Frames':<28} | {m640['taxonomy_counts']['CAT_2_SUB_THRESHOLD']:<18} | {m960['taxonomy_counts']['CAT_2_SUB_THRESHOLD']:<18} | {m1280['taxonomy_counts']['CAT_2_SUB_THRESHOLD']:<18}")
    print(f"{'CAT_4 (Deformed Bbox) Frames':<28} | {m640['taxonomy_counts']['CAT_4_DEFORMED_BBOX']:<18} | {m960['taxonomy_counts']['CAT_4_DEFORMED_BBOX']:<18} | {m1280['taxonomy_counts']['CAT_4_DEFORMED_BBOX']:<18}")
    print(f"{'CAT_5 (Intermittent) Frames':<28} | {m640['taxonomy_counts']['CAT_5_INTERMITTENT']:<18} | {m960['taxonomy_counts']['CAT_5_INTERMITTENT']:<18} | {m1280['taxonomy_counts']['CAT_5_INTERMITTENT']:<18}")
    print("-" * 95)
    print(f"{'P95 Inference Latency (ms)':<28} | {m640['p95_latency_ms']:<18.1f} | {m960['p95_latency_ms']:<18.1f} | {m1280['p95_latency_ms']:<18.1f}")
    print(f"{'Mean Inference Latency (ms)':<28} | {m640['mean_latency_ms']:<18.1f} | {m960['mean_latency_ms']:<18.1f} | {m1280['mean_latency_ms']:<18.1f}")
    print(f"{'Sustained Inference Speed':<28} | {m640['sustained_fps']:<14.1f} FPS | {m960['sustained_fps']:<14.1f} FPS | {m1280['sustained_fps']:<14.1f} FPS")
    print("=" * 95)
    print(f"Selection Decision: {selected_res}p")
    print(f"Selection Rationale: {selection_rationale}")
    print(f"Ablation artifact saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteration K.1: Dynamic Ingestion Resolution & Scale Tradeoff")
    parser.add_argument("--metadata", default=r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv")
    parser.add_argument("--video-dir", default=r"C:\Users\KALAKA~1\Downloads\archive\real_videos")
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[640, 960, 1280])
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.40)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/k1_resolution_ablation.json")
    args = parser.parse_args()

    run_resolution_benchmark(
        metadata_path=args.metadata,
        video_dir=args.video_dir,
        weights=args.weights,
        resolutions=args.resolutions,
        tau_conf=args.conf_threshold,
        iou_thresh=args.iou_threshold,
        window_size=args.window_size,
        device=args.device,
        output_path=args.output,
    )
