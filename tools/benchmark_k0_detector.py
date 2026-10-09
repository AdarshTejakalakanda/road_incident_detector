"""Iteration K.0: Frozen Baseline Instrumentation & Failure Taxonomy Harness.

Objective:
Isolate YOLO26 from the tracker and downstream logic. Establish ground-truth
bounding box telemetry and classify perception failures into deterministic buckets
around the incident window [t_GT - 30, t_GT + 30].

Failure Taxonomy Enum:
- CAT_1_NO_DETECTION: Detector output contains zero bboxes overlapping GT (IoU = 0).
- CAT_2_SUB_THRESHOLD: Bounding box exists (IoU >= 0.40), but confidence < tau_conf.
- CAT_3_WRONG_CLASS: Bounding box exists (IoU >= 0.40, confidence >= tau_conf), but class is non-vehicle.
- CAT_4_DEFORMED_BBOX: Detection exists with correct class, but IoU < 0.40 due to glare or clipping.
- CAT_5_INTERMITTENT: Detection flickers (<= 3 hits over 10 consecutive frames).
- SUCCESS: Valid detection hit (IoU >= 0.40, confidence >= tau_conf, correct vehicle class).

Outputs:
- artifacts/k0_baseline_taxonomy.json
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


class FailureTaxonomy(str, Enum):
    """Failure Taxonomy Categories per Iteration K.0 Specification."""
    SUCCESS = "SUCCESS"
    CAT_1_NO_DETECTION = "CAT_1_NO_DETECTION"
    CAT_2_SUB_THRESHOLD = "CAT_2_SUB_THRESHOLD"
    CAT_3_WRONG_CLASS = "CAT_3_WRONG_CLASS"
    CAT_4_DEFORMED_BBOX = "CAT_4_DEFORMED_BBOX"
    CAT_5_INTERMITTENT = "CAT_5_INTERMITTENT"


VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van", "train"}


def compute_iou_xyxy(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    ix1 = max(float(boxA[0]), float(boxB[0]))
    iy1 = max(float(boxA[1]), float(boxB[1]))
    ix2 = min(float(boxA[2]), float(boxB[2]))
    iy2 = min(float(boxA[3]), float(boxB[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    areaB = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    return float(inter / union) if union > 1e-6 else 0.0


def load_30_benchmark_entries(metadata_path: str) -> List[Dict[str, Any]]:
    """Load the standardized 30 benchmark video entries from split caches or metadata."""
    import pickle

    entries = []
    # Check for existing iteration B split cache pickles (15 in_distribution + 15 geo_aware)
    cached_splits = [
        ("split_in_distribution", "cache_videos_iter_b_split_in_distribution_15.pkl"),
        ("split_geo_aware", "cache_videos_iter_b_split_geo_aware_15.pkl"),
    ]

    used_from_cache = True
    for split_name, pickle_file in cached_splits:
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                for item in data:
                    entries.append({
                        "video_id": item["video_id"],
                        "gt_frame": int(item["gt_frame"]),
                        "split_type": split_name,
                    })
            except Exception as e:
                print(f"Warning: Failed loading {pickle_file}: {e}")
                used_from_cache = False
                break
        else:
            used_from_cache = False
            break

    if not used_from_cache or len(entries) != 30:
        print("Loading directly from metadata-real.csv...")
        entries.clear()
        loader = GroundTruthLoader(metadata_path)
        in_dist = loader.filter_split("test", "split_in_distribution")[:15]
        geo = loader.filter_split("test", "split_geo_aware")[:15]
        for r in in_dist:
            entries.append({"video_id": r.video_id, "gt_frame": r.accident_frame, "split_type": "split_in_distribution"})
        for r in geo:
            entries.append({"video_id": r.video_id, "gt_frame": r.accident_frame, "split_type": "split_geo_aware"})

    return entries


def evaluate_video_detections(
    video_path: str,
    gt_frame: int,
    gt_bbox_normalized: Tuple[float, float, float, float],
    model: YOLO,
    tau_conf: float = 0.25,
    iou_thresh: float = 0.40,
    window_half_size: int = 30,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate YOLO26 raw detections against ground truth across [t_GT - 30, t_GT + 30]."""
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
    eval_frame_indices = list(range(start_frame, end_frame + 1))

    # Scale normalized GT bounding box to pixel coordinates
    # gt_bbox_normalized: (x1, y1, x2, y2)
    gx1 = gt_bbox_normalized[0] * w
    gy1 = gt_bbox_normalized[1] * h
    gx2 = gt_bbox_normalized[2] * w
    gy2 = gt_bbox_normalized[3] * h
    gt_bbox_px = np.array([gx1, gy1, gx2, gy2], dtype=np.float32)

    gt_area_px = max(1.0, (gx2 - gx1) * (gy2 - gy1))
    norm_area = max(1e-6, (gt_bbox_normalized[2] - gt_bbox_normalized[0]) * (gt_bbox_normalized[3] - gt_bbox_normalized[1]))
    # COCO small object definition: area < 32^2 = 1024 px^2 or norm_area < 0.002
    is_small_object = bool(gt_area_px < 1024.0 or norm_area < 0.002)

    # Frame seek and inference
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    frame_telemetry = []
    inference_times = []

    # Sequential frame evaluation
    curr_f = start_frame - 1
    while curr_f < end_frame:
        ret, frame_img = cap.read()
        if not ret or frame_img is None:
            break
        curr_f += 1

        t_infer_start = time.perf_counter()
        # Run YOLO with low confidence floor to capture sub-threshold detections
        results = model.predict(source=frame_img, conf=0.01, device=device, verbose=False)
        t_infer_elapsed = time.perf_counter() - t_infer_start
        inference_times.append(t_infer_elapsed)

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

        # Determine frame-level detection category (before temporal flicker check)
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
            "inference_time_s": t_infer_elapsed,
        })

    cap.release()

    # Apply CAT_5_INTERMITTENT check (<= 3 hits over 10 consecutive frames)
    n_frames = len(frame_telemetry)
    hit_flags = [f["has_hit"] for f in frame_telemetry]

    for i in range(n_frames):
        # 10-frame window centered or local: [max(0, i-4), min(n_frames, i+6)]
        w_start = max(0, i - 4)
        w_end = min(n_frames, w_start + 10)
        if w_end - w_start < 10 and w_start > 0:
            w_start = max(0, w_end - 10)
        window_hits = sum(hit_flags[w_start:w_end])

        # If this frame was marked as SUCCESS but is in a sparse window (<= 3 hits over 10 frames)
        if frame_telemetry[i]["category"] == FailureTaxonomy.SUCCESS.value:
            if 0 < window_hits <= 3:
                frame_telemetry[i]["category"] = FailureTaxonomy.CAT_5_INTERMITTENT.value

    # Aggregate counts across evaluated window
    cat_counts = {cat.value: 0 for cat in FailureTaxonomy}
    for f in frame_telemetry:
        cat_counts[f["category"]] += 1

    total_eval = len(frame_telemetry)
    valid_hits = cat_counts[FailureTaxonomy.SUCCESS.value]
    vehicle_recall = valid_hits / total_eval if total_eval > 0 else 0.0

    # Determine dominant primary failure category for the incident
    if vehicle_recall >= 0.70:
        primary_cat = FailureTaxonomy.SUCCESS.value
    elif cat_counts[FailureTaxonomy.CAT_5_INTERMITTENT.value] > 0 and vehicle_recall < 0.40:
        primary_cat = FailureTaxonomy.CAT_5_INTERMITTENT.value
    else:
        # Pick the most frequent failure category
        fail_modes = [
            FailureTaxonomy.CAT_1_NO_DETECTION.value,
            FailureTaxonomy.CAT_2_SUB_THRESHOLD.value,
            FailureTaxonomy.CAT_3_WRONG_CLASS.value,
            FailureTaxonomy.CAT_4_DEFORMED_BBOX.value,
            FailureTaxonomy.CAT_5_INTERMITTENT.value,
        ]
        primary_cat = max(fail_modes, key=lambda m: cat_counts[m])

    avg_fps = (total_eval / sum(inference_times)) if inference_times and sum(inference_times) > 0 else 0.0

    return {
        "video_path": video_path,
        "total_frames_in_video": total_frames,
        "window_range": [start_frame, end_frame],
        "total_eval_frames": total_eval,
        "gt_frame": gt_frame,
        "gt_bbox_normalized": [round(float(x), 5) for x in gt_bbox_normalized],
        "gt_bbox_px": [round(float(x), 1) for x in gt_bbox_px],
        "gt_area_px": round(gt_area_px, 1),
        "is_small_object": is_small_object,
        "valid_hits": valid_hits,
        "vehicle_recall": round(vehicle_recall, 4),
        "primary_failure_category": primary_cat,
        "taxonomy_counts": cat_counts,
        "avg_fps": round(avg_fps, 2),
        "frame_telemetry": frame_telemetry,
    }


def main():
    parser = argparse.ArgumentParser(description="Iteration K.0: Frozen Baseline Instrumentation & Failure Taxonomy Harness")
    parser.add_argument("--metadata", default=r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv", help="Path to metadata-real.csv")
    parser.add_argument("--video-dir", default=r"C:\Users\KALAKA~1\Downloads\archive\real_videos", help="Path to video archive")
    parser.add_argument("--weights", default="yolo26n.pt", help="Path to YOLO weights")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Baseline confidence threshold tau_conf")
    parser.add_argument("--iou-threshold", type=float, default=0.40, help="IoU threshold for valid hit / failure taxonomy")
    parser.add_argument("--window-size", type=int, default=30, help="Half-window size around GT frame")
    parser.add_argument("--output", default="artifacts/k0_baseline_taxonomy.json", help="Output JSON path")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device")
    args = parser.parse_args()

    print("=" * 85)
    print("ITERATION K.0: FROZEN BASELINE INSTRUMENTATION & FAILURE TAXONOMY HARNESS")
    print("=" * 85)
    print(f"Metadata path:      {args.metadata}")
    print(f"Video archive dir:  {args.video_dir}")
    print(f"Model checkpoint:   {args.weights}")
    print(f"Confidence floor:   tau_conf = {args.conf_threshold}")
    print(f"IoU match thresh:   {args.iou_threshold}")
    print(f"Incident window:    [t_GT - {args.window_size}, t_GT + {args.window_size}]")
    print(f"Device:             {args.device}")
    print(f"Output artifact:    {args.output}")
    print("-" * 85)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("Loading Ground Truth and 30 Benchmark Video Entries...")
    loader = GroundTruthLoader(args.metadata)
    benchmark_entries = load_30_benchmark_entries(args.metadata)
    print(f"Loaded {len(benchmark_entries)} benchmark video entries.")

    print(f"Initializing YOLO26 from '{args.weights}'...")
    model = YOLO(args.weights)

    per_incident_results = []
    total_eval_frames_all = 0
    total_hits_all = 0
    small_hits_all = 0
    small_frames_all = 0
    norm_hits_all = 0
    norm_frames_all = 0
    total_inference_time_all = 0.0

    taxonomy_totals = {cat.value: 0 for cat in FailureTaxonomy}

    print("\nStarting Benchmark Run across 30 Incidents...")
    print(f"{'#':<3} | {'Video ID':<16} | {'Split':<12} | {'GT_f':<5} | {'Small?':<6} | {'Hits/Total':<11} | {'Recall':<7} | {'Primary Failure Category'}")
    print("-" * 95)

    # Cache inference for unique videos to ensure fast execution
    unique_cache: Dict[str, Dict[str, Any]] = {}

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

        # Evaluate or retrieve from cache
        cache_key = f"{vid}_{gt_f}"
        if cache_key in unique_cache:
            eval_res = unique_cache[cache_key]
        else:
            eval_res = evaluate_video_detections(
                video_path=vid_path,
                gt_frame=gt_f,
                gt_bbox_normalized=gt_record.bbox_normalized,
                model=model,
                tau_conf=args.conf_threshold,
                iou_thresh=args.iou_threshold,
                window_half_size=args.window_size,
                device=args.device,
            )
            unique_cache[cache_key] = eval_res

        if "error" in eval_res:
            print(f"Error evaluating {vid}: {eval_res['error']}")
            continue

        # Merge incident context
        eval_record = {
            "index": idx + 1,
            "video_id": vid,
            "split_type": split_t,
            "incident_type": gt_record.incident_type,
            "day_time": gt_record.day_time,
            "weather": gt_record.weather,
            "quality": gt_record.quality,
            "gt_frame": gt_f,
            "gt_bbox_normalized": eval_res["gt_bbox_normalized"],
            "gt_bbox_px": eval_res["gt_bbox_px"],
            "gt_area_px": eval_res["gt_area_px"],
            "is_small_object": eval_res["is_small_object"],
            "window_range": eval_res["window_range"],
            "total_eval_frames": eval_res["total_eval_frames"],
            "valid_hits": eval_res["valid_hits"],
            "vehicle_recall": eval_res["vehicle_recall"],
            "primary_failure_category": eval_res["primary_failure_category"],
            "taxonomy_counts": eval_res["taxonomy_counts"],
            "avg_fps": eval_res["avg_fps"],
            "frame_telemetry": eval_res["frame_telemetry"],
        }
        per_incident_results.append(eval_record)

        # Accumulate metrics
        n_f = eval_res["total_eval_frames"]
        n_h = eval_res["valid_hits"]
        total_eval_frames_all += n_f
        total_hits_all += n_h

        if eval_res["is_small_object"]:
            small_hits_all += n_h
            small_frames_all += n_f
        else:
            norm_hits_all += n_h
            norm_frames_all += n_f

        for cat, c_cnt in eval_res["taxonomy_counts"].items():
            taxonomy_totals[cat] += c_cnt

        for ft in eval_res["frame_telemetry"]:
            total_inference_time_all += ft["inference_time_s"]

        small_str = "YES" if eval_res["is_small_object"] else "NO"
        hits_str = f"{n_h}/{n_f}"
        rec_str = f"{eval_res['vehicle_recall']:.1%}"
        split_short = "in_dist" if "in_distribution" in split_t else "geo_aware"
        print(f"{idx+1:<3} | {vid:<16} | {split_short:<12} | {gt_f:<5} | {small_str:<6} | {hits_str:<11} | {rec_str:<7} | {eval_res['primary_failure_category']}")

    # Compute overall Agent Gate metrics
    raw_vehicle_recall = total_hits_all / total_eval_frames_all if total_eval_frames_all > 0 else 0.0
    small_object_recall = small_hits_all / small_frames_all if small_frames_all > 0 else 0.0
    normal_object_recall = norm_hits_all / norm_frames_all if norm_frames_all > 0 else 0.0
    baseline_fps = total_eval_frames_all / total_inference_time_all if total_inference_time_all > 0 else 0.0

    total_failures = total_eval_frames_all - taxonomy_totals[FailureTaxonomy.SUCCESS.value]
    taxonomy_percentages = {}
    for cat, cnt in taxonomy_totals.items():
        if cat == FailureTaxonomy.SUCCESS.value:
            taxonomy_percentages[cat] = round((cnt / total_eval_frames_all) * 100, 2) if total_eval_frames_all > 0 else 0.0
        else:
            taxonomy_percentages[cat] = round((cnt / total_failures) * 100, 2) if total_failures > 0 else 0.0

    summary_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_weights": args.weights,
            "tau_conf": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "window_size": args.window_size,
            "device": args.device,
            "total_benchmark_runs": len(per_incident_results),
            "unique_videos_evaluated": len(unique_cache),
        },
        "aggregate_metrics": {
            "raw_vehicle_recall": round(raw_vehicle_recall, 4),
            "small_object_recall": round(small_object_recall, 4),
            "medium_large_object_recall": round(normal_object_recall, 4),
            "baseline_fps": round(baseline_fps, 2),
            "total_frames_evaluated": total_eval_frames_all,
            "total_valid_hits": total_hits_all,
            "total_failure_frames": total_failures,
            "taxonomy_counts": taxonomy_totals,
            "taxonomy_percentages": taxonomy_percentages,
        },
        "per_incident_results": per_incident_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 85)
    print("ITERATION K.0: AGENT GATE BASELINE METRICS")
    print("=" * 85)
    print(f"Total Videos Evaluated:       {len(per_incident_results)}")
    print(f"Total Incident Frames:        {total_eval_frames_all}")
    print(f"Baseline YOLO26 Speed:        {baseline_fps:.1f} FPS")
    print(f"Raw Vehicle Recall:           {raw_vehicle_recall:.1%} ({total_hits_all}/{total_eval_frames_all})")
    print(f"Small-Object Recall:          {small_object_recall:.1%} ({small_hits_all}/{small_frames_all})")
    print(f"Medium/Large-Object Recall:   {normal_object_recall:.1%} ({norm_hits_all}/{norm_frames_all})")
    print("-" * 85)
    print("Failure Mode Distribution across Failure Frames:")
    for cat in [
        FailureTaxonomy.CAT_1_NO_DETECTION,
        FailureTaxonomy.CAT_2_SUB_THRESHOLD,
        FailureTaxonomy.CAT_3_WRONG_CLASS,
        FailureTaxonomy.CAT_4_DEFORMED_BBOX,
        FailureTaxonomy.CAT_5_INTERMITTENT,
    ]:
        cnt = taxonomy_totals[cat.value]
        pct = taxonomy_percentages[cat.value]
        print(f"  * {cat.value:<22}: {cnt:<5} frames ({pct:5.1f}% of failures)")
    print("=" * 85)
    print(f"Artifact successfully written to: {args.output}")


if __name__ == "__main__":
    main()
