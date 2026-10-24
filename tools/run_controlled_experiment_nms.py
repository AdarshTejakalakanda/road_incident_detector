"""Controlled Experiment 1: Class-Agnostic NMS (IoU=0.50) Evaluation on yzNEliTlFGk_00.

Protocol Requirements:
1. Isolate benchmark adapter only (no production changes, no tracker modifications).
2. Compare Pre-NMS vs Post-NMS outputs:
   - Raw duplicate box suppression count
   - Verification that distinct nearby vehicles are NOT merged
   - Evaluation of cross-class suppression
3. Run the exact same ByteTrack tracker pipeline on both streams:
   - Compare total track IDs created
   - Compare track fragmentation & lifespan
   - Compare active tracks during collision window (F40-F65)
   - Compare raw collision candidate proposals (especially IoU ~ 1.0 candidates)
   - Compare escalated alerts (Type 3 vs Type 5)
4. Generate side-by-side comparison video (Pre-NMS on left, Post-NMS on right) into output/annotated_videos.
"""

import os
import sys
import time
import json
import csv
import torch
import torchvision.ops as ops
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_threshold_verification_harness import (
    load_adapter,
    init_pipeline,
    resolve_video_path,
)
from road_surveillance.config import load_config
from road_surveillance.detection.detector_interface import DetectionBatch
from modules.telemetry_accumulator import TelemetryAccumulator
from modules.episode_consolidator import consolidate_candidates_into_episodes
from modules.scene_escalator import SceneEscalator


def apply_class_agnostic_nms(out, iou_thresh: float = 0.50):
    """Applies class-agnostic NMS to StandardDetectionOutput."""
    if len(out.boxes) <= 1:
        return out, 0
    
    t_boxes = torch.from_numpy(out.boxes)
    t_scores = torch.from_numpy(out.scores)
    keep = ops.nms(t_boxes, t_scores, iou_threshold=iou_thresh).numpy()
    suppressed_count = len(out.boxes) - len(keep)

    from tools.detector_ab_adapters import StandardDetectionOutput
    post_out = StandardDetectionOutput(
        frame_id=out.frame_id,
        boxes=out.boxes[keep],
        scores=out.scores[keep],
        class_ids=out.class_ids[keep],
        class_names=[out.class_names[i] for i in keep],
        unified_class_ids=out.unified_class_ids[keep] if len(out.unified_class_ids) > 0 else np.empty((0,), dtype=int),
        unified_class_names=[out.unified_class_names[i] for i in keep],
        mapping_version=out.mapping_version,
    )
    return post_out, suppressed_count


def run_pipeline(frames, detections_list, fps, w, h, clip_id, default_calib, tau_esc=0.50, mode_name="Baseline"):
    """Runs identical tracking, event engine, and scene escalator on a detection stream."""
    tracker, geom_engine, event_engine, lane_manager = init_pipeline(
        w, h, fps, conf_th=0.25, default_calib_cfg=default_calib
    )
    telemetry_acc = TelemetryAccumulator()
    scene_escalator = SceneEscalator(tau_aftermath=tau_esc)

    all_candidates = []
    frame_annotated = []
    active_tracks_by_frame = {}
    track_lifespans = {}

    for f_idx, (frame, out) in enumerate(zip(frames, detections_list)):
        pts_ns = int(f_idx * (1e9 / fps))
        timestamp_s = f_idx / fps

        batch = DetectionBatch(
            video_id=clip_id,
            frame_idx=f_idx,
            pts_ns=pts_ns,
            boxes=out.boxes,
            confidences=out.scores,
            class_ids=out.unified_class_ids if len(out.unified_class_ids) > 0 else np.empty((0,), dtype=int),
            class_labels=out.unified_class_names,
        )
        tracklets, active_tracks = tracker.update(batch)
        kinematics, convergences = geom_engine.process_frame(tracklets, active_tracks)
        
        telemetry_acc.record_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            timestamp_s=timestamp_s,
            active_tracks=active_tracks,
            kinematics_map=kinematics,
        )

        cands = event_engine.process_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            video_id=clip_id,
            tracks=active_tracks,
            kinematics=kinematics,
            convergences=convergences,
            lane_manager=lane_manager,
        )

        for c in cands:
            etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
            all_candidates.append({
                "frame": f_idx,
                "timestamp_s": round(timestamp_s, 3),
                "event_type": etype,
                "track_ids": c.track_ids,
                "confidence": round(float(c.confidence), 2),
                "description": c.description,
                "signals": c.signals or {},
            })

        active_tracks_by_frame[f_idx] = len(active_tracks)
        for tid, tr in active_tracks.items():
            if tid not in track_lifespans:
                track_lifespans[tid] = {"first": f_idx, "last": f_idx, "count": 0, "class": tr.class_label}
            track_lifespans[tid]["last"] = f_idx
            track_lifespans[tid]["count"] += 1

        # Annotation layer
        ann = frame.copy()
        for bx, sc, un in zip(out.boxes, out.scores, out.unified_class_names):
            x1, y1, x2, y2 = map(int, bx)
            color = (0, 0, 255) if mode_name == "Pre-NMS (Raw)" else (0, 255, 0)
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            cv2.putText(ann, f"{un} {sc:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for tid, tr in active_tracks.items():
            bx, by, bw, bh = map(int, tr.current_bbox_xywh)
            cv2.putText(ann, f"ID:{tid}", (bx, by + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(ann, f"[{mode_name}] Frame: {f_idx:03d} | Active IDs: {len(active_tracks)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        frame_annotated.append(ann)

    episodes = consolidate_candidates_into_episodes(
        raw_candidates=all_candidates,
        video_id=clip_id,
        cooldown_seconds=10.0,
        max_gap_frames=60,
    )

    escalated_alerts = []
    duration_s = len(frames) / fps
    for ep in episodes:
        t_anchor = float(ep.peak_candidate["timestamp_s"])
        t_end = min(t_anchor + 1.50, duration_s)
        window_history = telemetry_acc.get_window(t_start=t_anchor, t_end=t_end)
        res = scene_escalator.evaluate_candidate(
            candidate=ep.peak_candidate,
            track_history=window_history,
            fps=fps,
        )
        if res.score is not None and res.score >= tau_esc:
            escalated_alerts.append({
                "frame": int(ep.peak_candidate["frame"]),
                "timestamp_s": round(t_anchor, 3),
                "type": res.canonical_type,
                "score": round(float(res.score), 4),
                "channels": res.channels,
            })

    return {
        "annotated_frames": frame_annotated,
        "track_lifespans": track_lifespans,
        "all_candidates": all_candidates,
        "escalated_alerts": escalated_alerts,
        "active_tracks_by_frame": active_tracks_by_frame,
    }


def run_experiment():
    video_path = resolve_video_path("yzNEliTlFGk_00")
    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.abspath("output/annotated_videos")
    os.makedirs(out_dir, exist_ok=True)

    print("=================================================================")
    print("CONTROLLED EXPERIMENT 1: CLASS-AGNOSTIC NMS (IoU = 0.50)")
    print(f"Target Video: {clip_id} | Model: Cand_B (UVH-26 RT-DETRv2-S)")
    print(f"Confidence: tau_conf = 0.25 | Escalation: tau_esc = 0.50")
    print(f"Tracker: ByteTrack (Frozen / Baseline parameters)")
    print("=================================================================\n")

    cfg = load_config()
    default_calib = cfg.calibration.get("default")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.033
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()
    total_frames = len(raw_frames)
    duration_s = total_frames / fps
    print(f"Preloaded {total_frames} frames ({w}x{h} @ {fps:.2f} fps, {duration_s:.2f}s)\n")

    adapter = load_adapter("Cand_B")

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = adapter.detect(dummy, conf_threshold=0.25, frame_id=-1)

    print("Step 1.1: Extracting raw pre-NMS detections (tau_conf = 0.25)...")
    pre_nms_detections = []
    post_nms_detections = []
    total_raw_boxes = 0
    total_suppressed_boxes = 0
    frames_with_suppression = 0

    # Cross-class tracking
    cross_class_suppressions = []

    for f_idx, frame in enumerate(raw_frames):
        out = adapter.detect(frame, conf_threshold=0.25, frame_id=f_idx)
        pre_nms_detections.append(out)
        total_raw_boxes += len(out.boxes)

        post_out, suppressed = apply_class_agnostic_nms(out, iou_thresh=0.50)
        post_nms_detections.append(post_out)
        total_suppressed_boxes += suppressed
        if suppressed > 0:
            frames_with_suppression += 1

        # Check for cross-class collisions in raw
        if len(out.boxes) > 1:
            t_boxes = torch.from_numpy(out.boxes)
            t_scores = torch.from_numpy(out.scores)
            ious = ops.box_iou(t_boxes, t_boxes).numpy()
            for i in range(len(out.boxes)):
                for j in range(i + 1, len(out.boxes)):
                    if ious[i, j] >= 0.50:
                        c1 = out.unified_class_names[i]
                        c2 = out.unified_class_names[j]
                        s1 = out.scores[i]
                        s2 = out.scores[j]
                        cross_class_suppressions.append((f_idx, c1, s1, c2, s2, ious[i, j]))

    del adapter

    total_post_boxes = total_raw_boxes - total_suppressed_boxes
    pct_suppressed = (total_suppressed_boxes / total_raw_boxes) * 100 if total_raw_boxes > 0 else 0

    print(f"\n--- Detection Filtering Results ---")
    print(f"Total Raw Pre-NMS Bounding Boxes:  {total_raw_boxes} ({total_raw_boxes/total_frames:.2f} boxes/frame)")
    print(f"Total Duplicate Boxes Suppressed:   {total_suppressed_boxes} ({pct_suppressed:.1f}% reduction)")
    print(f"Total Clean Post-NMS Bounding Boxes: {total_post_boxes} ({total_post_boxes/total_frames:.2f} boxes/frame)")
    print(f"Frames affected by duplicates:       {frames_with_suppression}/{total_frames} ({(frames_with_suppression/total_frames)*100:.1f}%)")
    print(f"Overlapping cross-box instances:    {len(cross_class_suppressions)}")

    print("\nStep 1.2: Running Baseline Pipeline (Pre-NMS stream)...")
    res_pre = run_pipeline(raw_frames, pre_nms_detections, fps, w, h, clip_id, default_calib, tau_esc=0.50, mode_name="Pre-NMS (Raw)")

    print("Step 1.3: Running Experiment 1 Pipeline (Post-NMS @ IoU=0.50 stream)...")
    res_post = run_pipeline(raw_frames, post_nms_detections, fps, w, h, clip_id, default_calib, tau_esc=0.50, mode_name="Post-NMS (IoU=0.50)")

    # Quantitative Comparison
    track_ids_pre = len(res_pre["track_lifespans"])
    track_ids_post = len(res_post["track_lifespans"])

    cands_pre = len(res_pre["all_candidates"])
    cands_post = len(res_post["all_candidates"])

    zero_dist_cands_pre = sum(1 for c in res_pre["all_candidates"] if c.get("signals", {}).get("min_distance_m", 99) < 0.1 or c.get("signals", {}).get("bbox_iou", 0) > 0.90)
    zero_dist_cands_post = sum(1 for c in res_post["all_candidates"] if c.get("signals", {}).get("min_distance_m", 99) < 0.1 or c.get("signals", {}).get("bbox_iou", 0) > 0.90)

    alerts_pre = res_pre["escalated_alerts"]
    alerts_post = res_post["escalated_alerts"]

    # Render Side-by-Side Comparison Video
    print("\nStep 1.4: Rendering Side-by-Side Comparison Video...")
    side_by_side_mp4 = os.path.join(out_dir, f"{clip_id}_PRE_vs_POST_NMS_comparison.mp4")
    side_by_side_gif = os.path.join(out_dir, f"{clip_id}_PRE_vs_POST_NMS_comparison.gif")

    writer_sbs = cv2.VideoWriter(side_by_side_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    sbs_gif_frames = []

    for f_idx in range(total_frames):
        left = res_pre["annotated_frames"][f_idx]
        right = res_post["annotated_frames"][f_idx]
        sbs_frame = np.hstack([left, right])
        writer_sbs.write(sbs_frame)

        # GIF sample (every 2nd frame or half scale for reasonable size)
        rgb = cv2.cvtColor(sbs_frame, cv2.COLOR_BGR2RGB)
        gh, gw = rgb.shape[:2]
        resized = cv2.resize(rgb, (gw // 4, gh // 4), interpolation=cv2.INTER_AREA)
        sbs_gif_frames.append(Image.fromarray(resized))

    writer_sbs.release()
    print(f"  -> Saved Side-by-Side MP4: {side_by_side_mp4}")

    if sbs_gif_frames:
        sbs_gif_frames[0].save(
            side_by_side_gif,
            save_all=True,
            append_images=sbs_gif_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"  -> Saved Side-by-Side GIF: {side_by_side_gif}")

    # Also save clean post-NMS full video
    post_full_mp4 = os.path.join(out_dir, f"Cand_B_tau0_25_{clip_id}_POST_NMS_FULL_annotated.mp4")
    post_full_gif = os.path.join(out_dir, f"Cand_B_tau0_25_{clip_id}_POST_NMS_FULL_annotated.gif")
    writer_post = cv2.VideoWriter(post_full_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    gif_frames_clean = []
    for f in res_post["annotated_frames"]:
        writer_post.write(f)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        gh, gw = rgb.shape[:2]
        gif_frames_clean.append(Image.fromarray(cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)))
    writer_post.release()
    if gif_frames_clean:
        gif_frames_clean[0].save(post_full_gif, save_all=True, append_images=gif_frames_clean[1:], duration=int(1000 / fps), loop=0)

    # Save summary report JSON
    report = {
        "clip_id": clip_id,
        "model": "Cand_B (UVH-26 RT-DETRv2-S)",
        "tau_conf": 0.25,
        "tau_esc": 0.50,
        "metrics": {
            "total_detections_pre_nms": total_raw_boxes,
            "total_detections_post_nms": total_post_boxes,
            "duplicate_boxes_suppressed": total_suppressed_boxes,
            "suppression_percentage": round(pct_suppressed, 1),
            "track_ids_created_pre_nms": track_ids_pre,
            "track_ids_created_post_nms": track_ids_post,
            "track_id_reduction": track_ids_pre - track_ids_post,
            "raw_candidates_pre_nms": cands_pre,
            "raw_candidates_post_nms": cands_post,
            "zero_dist_phantom_candidates_pre_nms": zero_dist_cands_pre,
            "zero_dist_phantom_candidates_post_nms": zero_dist_cands_post,
            "escalated_alerts_pre_nms": len(alerts_pre),
            "escalated_alerts_post_nms": len(alerts_post),
            "alerts_pre_details": alerts_pre,
            "alerts_post_details": alerts_post,
        }
    }
    report_path = os.path.join(out_dir, f"{clip_id}_experiment_1_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n========================================================")
    print("CONTROLLED EXPERIMENT 1 COMPLETE")
    print(f"Report: {report_path}")
    print(f"Side-by-Side MP4: {side_by_side_mp4}")
    print(f"Side-by-Side GIF: {side_by_side_gif}")
    print("========================================================\n")

    return report


if __name__ == "__main__":
    run_experiment()
