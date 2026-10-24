"""Annotated Video Generation CLI for Detector A/B Comparison.

Allows running any or all of the 5 detector configurations (Arms A-E) on local CCTV videos
and saves fully-annotated MP4 recordings with bounding boxes, tracking trails, confidence scores,
and a diagnostic HUD overlay.

Usage Examples:
    # Run Arm A (production YOLO26n) on default sample video (first 150 frames):
    .venv\\Scripts\\python tools/run_annotated_detector.py --arm A --max-frames 150

    # Run Arm B (UVH-26 YOLO11-S) on a specific video:
    .venv\\Scripts\\python tools/run_annotated_detector.py --arm B --video 5uh8-3q51uI_00 --max-frames 150

    # Run all 5 arms to generate side-by-side comparative videos:
    .venv\\Scripts\\python tools/run_annotated_detector.py --arm all --max-frames 100
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.detector_ab_adapters import (
    MAPPING_VERSION,
    UNIFIED_TAXONOMY,
    UltralyticsYOLOAdapter,
    RTDETRv2Adapter,
    StandardDetectionOutput,
)
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch


# ---------------------------------------------------------------------------
# Visual Palette and HUD Styler
# ---------------------------------------------------------------------------

COLOR_PALETTE = [
    (255, 128, 0),    # Vivid Sky Blue
    (50, 205, 50),    # Lime Green
    (0, 165, 255),    # Amber Orange
    (238, 130, 238),  # Lavender Violet
    (0, 215, 255),    # Vibrant Yellow-Gold
    (255, 20, 147),   # Deep Pink
    (0, 255, 255),    # Cyber Yellow
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Magenta
    (180, 105, 255),  # Hot Pink
]

ARM_DESCRIPTIONS = {
    "A": ("Arm A: Production Baseline", "YOLO26n (COCO-80)"),
    "B": ("Arm B: UVH-26 Reference", "YOLO11-S (Bengaluru Safe City, 14 Classes)"),
    "C": ("Arm C: RT-DETRv2-S", "Transformer Detector (ResNet18vd, 15 Classes)"),
    "D": ("Arm D: Highway Traffic", "YOLOv8m (vietnguyen Highway, 8 Classes)"),
    "E": ("Arm E: Night Specialist", "YOLOv8s (G2mus Night-vehicle, 3 Classes)"),
}


def resolve_video_path(video_input: str) -> str:
    """Resolve file path from direct path, video ID, or dataset archive."""
    if os.path.isfile(video_input):
        return os.path.abspath(video_input)

    vid_base = os.path.basename(video_input)
    if not vid_base.endswith(".mp4"):
        vid_base = f"{vid_base}.mp4"

    candidates = [
        video_input,
        os.path.join(r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos", vid_base),
        os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", vid_base),
        os.path.join(os.path.expanduser("~"), "Downloads", "archive", "real_videos", vid_base),
        os.path.join("data", "videos", vid_base),
        os.path.join(".", vid_base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(f"Video '{video_input}' could not be located on disk.")


def get_adapter(arm: str):
    """Instantiate the requested detector adapter."""
    arm = arm.upper()
    if arm == "A":
        return UltralyticsYOLOAdapter(
            arm_id="Arm A",
            model_name="YOLO26n (Control)",
            weights_path="models/yolo26n.pt",
            unified_map={1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"},
        )
    elif arm == "B":
        return UltralyticsYOLOAdapter(
            arm_id="Arm B",
            model_name="UVH-26 YOLO11-S",
            weights_path="models/benchmark_arms/UVH-26-MV-YOLOv11-S.pt",
            unified_map={
                0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
                6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
                10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
            },
        )
    elif arm == "C":
        return RTDETRv2Adapter(
            arm_id="Arm C",
            model_name="UVH-26 RT-DETRv2-S",
            config_path="models/benchmark_arms/rtdetr_s_flattened.yaml",
            weights_path="models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth",
            unified_map={
                0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
                6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
                10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
            },
        )
    elif arm == "D":
        return UltralyticsYOLOAdapter(
            arm_id="Arm D",
            model_name="vietnguyen Highway YOLOv8m",
            weights_path="models/benchmark_arms/vietnguyen_best.pt",
            unified_map={
                0: "auto_rickshaw", 1: "bus", 2: "car", 3: "truck",
                4: "motorcycle", 5: "truck", 6: "other_vehicle", 7: "truck"
            },
        )
    elif arm == "E":
        return UltralyticsYOLOAdapter(
            arm_id="Arm E",
            model_name="G2mus Night YOLOv8s",
            weights_path="models/benchmark_arms/night-vehicle-yolov8s.pt",
            unified_map={0: "car", 1: "truck", 2: "bus"},
        )
    elif arm == "F":
        return UltralyticsYOLOAdapter(
            arm_id="Arm F",
            model_name="BDD100K Night YOLO11-S",
            weights_path="models/benchmark_arms/night_specialists/bdd100k_yolo11s.pt",
            unified_map={
                2: "car", 3: "truck", 4: "bus", 5: "truck", 6: "motorcycle", 7: "bicycle"
            },
        )
    else:
        raise ValueError(f"Unknown arm: {arm}. Choose from A, B, C, D, E, F, or all.")


def draw_annotation_layer(
    image: np.ndarray,
    out: StandardDetectionOutput,
    tracklets: list,
    tracks: dict,
    arm_key: str,
    frame_idx: int,
    fps_instant: float,
) -> np.ndarray:
    """Draw bounding boxes, track trails, class tags, and diagnostic HUD."""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # Map box coordinates to detections
    # Draw tracking boxes and trails if tracklets are available
    tracked_box_indices = set()
    for tracklet in tracklets:
        tid = tracklet.track_id
        color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]

        bx, by, bw, bh = tracklet.bbox_xywh
        x1, y1 = max(0, int(bx)), max(0, int(by))
        x2, y2 = min(w - 1, int(bx + bw)), min(h - 1, int(by + bh))

        # Main BBox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Ground contact circle
        gc_x, gc_y = int(tracklet.ground_contact_px[0]), int(tracklet.ground_contact_px[1])
        cv2.circle(canvas, (gc_x, gc_y), 4, (0, 0, 255), -1)
        cv2.circle(canvas, (gc_x, gc_y), 6, color, 1)

        # Trajectory trail
        track = tracks.get(tid)
        if track and len(track.position_history) >= 2:
            pts = np.array(track.position_history[-20:], dtype=np.int32)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

        # Label tag
        label_text = f"ID:{tid} {tracklet.class_label}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_y1 = max(0, y1 - th - 5)
        tag_y2 = y1
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 6, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label_text,
            (x1 + 3, tag_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0) if (color[0] + color[1] + color[2] > 400) else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Top-Left HUD Banner
    title, desc = ARM_DESCRIPTIONS.get(arm_key, (f"Arm {arm_key}", ""))
    hud_w = min(420, w - 20)
    hud_h = 75

    hud_sub = canvas[10:10 + hud_h, 10:10 + hud_w]
    black_rect = np.zeros_like(hud_sub)
    cv2.addWeighted(black_rect, 0.75, hud_sub, 0.25, 0, hud_sub)
    canvas[10:10 + hud_h, 10:10 + hud_w] = hud_sub

    # Outer border for HUD
    cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

    # HUD Texts
    cv2.putText(canvas, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, desc, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    status_line = f"Frame: {frame_idx:04d} | Dets: {len(out.boxes)} | Tracks: {len(tracks)} | FPS: {fps_instant:.1f}"
    cv2.putText(canvas, status_line, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

    return canvas


def process_video_arm(
    arm: str,
    video_path: str,
    output_dir: str,
    max_frames: int = 150,
    conf_threshold: float = 0.25,
) -> str:
    """Run specified arm on video and save annotated MP4."""
    arm = arm.upper()
    adapter = get_adapter(arm)
    resolved_path = resolve_video_path(video_path)
    video_id = os.path.splitext(os.path.basename(resolved_path))[0]

    os.makedirs(output_dir, exist_ok=True)
    out_filename = f"{arm}_{video_id}_annotated.mp4"
    out_path = os.path.join(output_dir, out_filename)

    cap = cv2.VideoCapture(resolved_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {resolved_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_to_run = min(total_video_frames, max_frames) if max_frames > 0 else total_video_frames

    print(f"\n========================================================")
    print(f"Running Arm [{arm}]: {adapter.model_name}")
    print(f"Source: {resolved_path} ({w}x{h} @ {fps_in:.1f} FPS)")
    print(f"Target: {frames_to_run} frames -> {out_path}")
    print(f"========================================================")

    # Initialize video writer (using mp4v codec for universal Windows playback)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    # Isolated ByteTrack tracker instance
    tracker = ByteTrackTracker(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps_in),
    )

    t_start = time.perf_counter()
    frame_count = 0
    total_detections = 0

    while cap.isOpened() and frame_count < frames_to_run:
        ret, frame = cap.read()
        if not ret:
            break

        t_frame0 = time.perf_counter()

        # 1. Run Detector Adapter
        out = adapter.detect(frame, frame_id=frame_count, conf_threshold=conf_threshold)
        total_detections += len(out.boxes)

        # 2. Update ByteTrack Tracker
        batch = DetectionBatch(
            video_id=video_id,
            frame_idx=frame_count,
            pts_ns=int(frame_count * (1e9 / fps_in)),
            boxes=out.boxes,
            confidences=out.scores,
            class_ids=out.unified_class_ids,
            class_labels=out.unified_class_names,
        )
        tracklets, tracks = tracker.update(batch)

        t_frame_end = time.perf_counter()
        fps_instant = 1.0 / max(1e-4, t_frame_end - t_frame0)

        # 3. Draw Complete Annotation Layer
        annotated_frame = draw_annotation_layer(
            image=frame,
            out=out,
            tracklets=tracklets,
            tracks=tracks,
            arm_key=arm,
            frame_idx=frame_count,
            fps_instant=fps_instant,
        )

        # 4. Write Frame
        writer.write(annotated_frame)
        frame_count += 1

        if frame_count % 25 == 0 or frame_count == frames_to_run:
            elapsed = time.perf_counter() - t_start
            avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
            print(f"  Frame {frame_count:03d}/{frames_to_run} ({frame_count/frames_to_run*100:5.1f}%) | "
                  f"Avg FPS: {avg_fps:.1f} | Active Tracks: {len(tracks):2d} | Total Dets: {total_detections:3d}")

    cap.release()
    writer.release()

    total_time = time.perf_counter() - t_start
    print(f"\n[Arm {arm}] Finished in {total_time:.2f}s ({frame_count/total_time:.1f} FPS)")
    print(f"Video Saved: {out_path} ({os.path.getsize(out_path):,} bytes)\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate Annotated Videos for Detector A/B Comparison")
    parser.add_argument("--arm", type=str, default="A", choices=["A", "B", "C", "D", "E", "all"],
                        help="Detector Arm to run (A: YOLO26n, B: UVH-YOLO11, C: RT-DETRv2, D: Highway, E: Night, all: Run all)")
    parser.add_argument("--video", type=str, default="5uh8-3q51uI_00",
                        help="Video filename, ID, or path (default: 5uh8-3q51uI_00)")
    parser.add_argument("--max-frames", type=int, default=150,
                        help="Maximum frames to annotate (default: 150 frames, ~5 seconds @ 30 FPS. Use 0 for full video)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", type=str, default="output/annotated_videos",
                        help="Output directory to save videos (default: output/annotated_videos)")

    args = parser.parse_args()

    if args.arm.lower() == "all":
        arms_to_run = ["A", "B", "C", "D", "E"]
    else:
        arms_to_run = [args.arm.upper()]

    generated_videos = []
    for arm in arms_to_run:
        path = process_video_arm(
            arm=arm,
            video_path=args.video,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            conf_threshold=args.conf,
        )
        generated_videos.append(path)

    print("========================================================")
    print("ALL ANNOTATED VIDEOS GENERATED SUCCESSFULLY:")
    for v in generated_videos:
        print(f"  -> {v}")
    print("========================================================")


if __name__ == "__main__":
    main()
