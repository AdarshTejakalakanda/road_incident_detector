"""Generate Full Continuous 30-Second Annotated Videos for Threshold Visual Verification."""

import os
import sys
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_threshold_verification_harness import (
    load_adapter,
    init_pipeline,
    generate_html_gallery,
    resolve_video_path,
    ARM_MAP,
)
from road_surveillance.config import load_config
from road_surveillance.detection.detector_interface import DetectionBatch


def generate_full_videos(
    video_path: str = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\yzNEliTlFGk_00.mp4",
    arm_key: str = "Cand_B",
    thresholds = [0.20, 0.25, 0.30],
    output_root: str = "evidence/threshold_sweep_yzNEliTlFGk",
):
    resolved_video = resolve_video_path(video_path)
    clip_id = os.path.splitext(os.path.basename(resolved_video))[0]
    pkg_dir = os.path.join(output_root, "packages")
    os.makedirs(pkg_dir, exist_ok=True)

    cfg = load_config()
    default_calib = cfg.calibration.get("default")

    cap = cv2.VideoCapture(resolved_video)
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

    arm_name = ARM_MAP[arm_key][0]
    print(f">>> Running Full Video Generation for {arm_key} ({arm_name})")
    adapter = load_adapter(arm_key)
    base_conf = min(thresholds)

    # Warmup
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(10):
        _ = adapter.detect(dummy, conf_threshold=base_conf, frame_id=-1)

    print(f"Extracting base logits at base_conf={base_conf}...")
    frame_detections = []
    for f_idx, frame in enumerate(raw_frames):
        out = adapter.detect(frame, conf_threshold=base_conf, frame_id=f_idx)
        frame_detections.append(out)
    del adapter

    for tau_c in thresholds:
        print(f"\n--- Generating Full Continuous Video for tau_conf = {tau_c:.2f} ---")
        tracker, geom_engine, event_engine, lane_manager = init_pipeline(
            w, h, fps, conf_th=tau_c, default_calib_cfg=default_calib
        )

        frame_annotated = []
        tau_str = f"{tau_c:.2f}".replace(".", "_")

        for f_idx, (frame, out_base) in enumerate(zip(raw_frames, frame_detections)):
            mask = out_base.scores >= tau_c
            b_boxes = out_base.boxes[mask]
            b_scores = out_base.scores[mask]
            b_u_ids = out_base.unified_class_ids[mask] if len(out_base.unified_class_ids) > 0 else np.empty((0,), dtype=int)
            b_u_names = [out_base.unified_class_names[i] for i, m in enumerate(mask) if m]

            pts_ns = int(f_idx * (1e9 / fps))
            timestamp_s = f_idx / fps

            batch = DetectionBatch(
                video_id=clip_id,
                frame_idx=f_idx,
                pts_ns=pts_ns,
                boxes=b_boxes,
                confidences=b_scores,
                class_ids=b_u_ids,
                class_labels=b_u_names,
            )
            tracklets, active_tracks = tracker.update(batch)

            ann_frame = frame.copy()
            for box, score, uname in zip(b_boxes, b_scores, b_u_names):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(ann_frame, f"{uname} {score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            for tid, tr in active_tracks.items():
                bx, by, bw, bh = map(int, tr.current_bbox_xywh)
                cv2.putText(ann_frame, f"ID:{tid}", (bx, by + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ann_frame, f"FULL VIDEO | Frame: {f_idx:03d}/240 | Time: {timestamp_s:.2f}s | {arm_key} @ tau={tau_c:.2f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            frame_annotated.append(ann_frame)

        # 1. Full continuous MP4
        full_base = f"{arm_key}_tau{tau_str}_{clip_id}_FULL"
        full_mp4_path = os.path.join(pkg_dir, f"{full_base}_annotated.mp4")
        writer_full = cv2.VideoWriter(full_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frame_annotated:
            writer_full.write(fr)
        writer_full.release()
        print(f"  -> Saved Full MP4: {full_mp4_path}")

        # 2. Full continuous GIF for direct browser viewing
        full_gif_path = os.path.join(pkg_dir, f"{full_base}_annotated.gif")
        gif_frames = []
        for fr in frame_annotated:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            gh, gw = rgb.shape[:2]
            resized = cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)
            gif_frames.append(Image.fromarray(resized))
        if gif_frames:
            gif_frames[0].save(
                full_gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=int(1000 / fps),
                loop=0
            )
            print(f"  -> Saved Full GIF: {full_gif_path}")

    # Refresh gallery
    generate_html_gallery(output_root, clip_id)
    print("\nFull videos generated successfully and gallery refreshed!")


if __name__ == "__main__":
    generate_full_videos()
