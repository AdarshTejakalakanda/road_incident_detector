"""Diagnose Perception Layer and ByteTrack ID fragmentation on yzNEliTlFGk_00."""

import os
import sys
import cv2
import numpy as np

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


def diagnose():
    video_path = resolve_video_path("yzNEliTlFGk_00")
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

    cfg = load_config()
    default_calib = cfg.calibration.get("default")

    adapter = load_adapter("Cand_B")
    print("Running Cand_B inference at tau=0.25...")
    
    # Run with default tracker
    tracker, geom, event_eng, lanes = init_pipeline(w, h, fps, conf_th=0.25, default_calib_cfg=default_calib)

    track_lifespans = {} # tid -> [first_f, last_f, total_detections, max_score, classes, bboxes]
    per_frame_boxes = []

    for f_idx, frame in enumerate(raw_frames):
        out = adapter.detect(frame, conf_threshold=0.25, frame_id=f_idx)
        per_frame_boxes.append(len(out.boxes))

        pts_ns = int(f_idx * (1e9 / fps))
        batch = DetectionBatch(
            video_id="yzNEliTlFGk_00",
            frame_idx=f_idx,
            pts_ns=pts_ns,
            boxes=out.boxes,
            confidences=out.scores,
            class_ids=out.unified_class_ids if len(out.unified_class_ids) > 0 else np.empty((0,), dtype=int),
            class_labels=out.unified_class_names,
        )
        tracklets, active_tracks = tracker.update(batch)

        for tid, tr in active_tracks.items():
            if tid not in track_lifespans:
                track_lifespans[tid] = {
                    "first_frame": f_idx,
                    "last_frame": f_idx,
                    "count": 0,
                    "class": tr.class_label,
                    "bboxes": [],
                }
            track_lifespans[tid]["last_frame"] = f_idx
            track_lifespans[tid]["count"] += 1
            track_lifespans[tid]["bboxes"].append((f_idx, tr.current_bbox_xywh))

    print(f"\nTotal Track IDs Created Across 240 Frames: {len(track_lifespans)}")
    print(f"Average Detections Per Frame: {np.mean(per_frame_boxes):.2f}")
    
    # Analyze track persistence
    single_frame_tracks = [tid for tid, d in track_lifespans.items() if d["count"] <= 2]
    long_tracks = [tid for tid, d in track_lifespans.items() if d["count"] >= 15]

    print(f"Ephemeral/Flickering Tracks (<= 2 frames): {len(single_frame_tracks)} ({(len(single_frame_tracks)/len(track_lifespans))*100:.1f}%)")
    print(f"Stable/Persistent Tracks (>= 15 frames): {len(long_tracks)}")

    print("\n--- Longest Lived Tracks (Potential Real Vehicles) ---")
    sorted_tracks = sorted(track_lifespans.items(), key=lambda x: x[1]["count"], reverse=True)
    for tid, d in sorted_tracks[:12]:
        dur_s = (d["last_frame"] - d["first_frame"]) / fps
        print(f"  Track ID {tid:2d} | Class: {d['class']:<8} | Frames {d['first_frame']:3d} -> {d['last_frame']:3d} ({dur_s:4.1f}s, {d['count']:3d} hits)")

    print("\n--- Tracks around Ground-Truth Collision (Frames 40 - 65) ---")
    collision_tracks = [
        (tid, d) for tid, d in track_lifespans.items()
        if not (d["last_frame"] < 35 or d["first_frame"] > 70)
    ]
    for tid, d in sorted(collision_tracks, key=lambda x: x[1]["first_frame"]):
        print(f"  Track ID {tid:2d} | Class: {d['class']:<8} | Active F{d['first_frame']:03d} -> F{d['last_frame']:03d} (Duration: {d['count']} frames)")


if __name__ == "__main__":
    diagnose()
