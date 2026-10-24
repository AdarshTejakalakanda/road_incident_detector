"""Configurable Threshold Sweep & Manual Verification Harness for Road Surveillance Incidents.

Executes a configurable threshold sweep across:
  - Arms: Control (YOLO26n), Cand_A (YOLO11-S), Cand_B (RT-DETRv2-S), Cand_C (Highway YOLOv8m), Cand_D (Night YOLOv8s)
  - tau_conf: any custom list of float thresholds (e.g., 0.15, 0.20, 0.25, 0.30, 0.35)
  - tau_esc: SceneEscalator composite threshold (default: 0.50)
  - Video: any video path or clip ID (e.g., yzNEliTlFGk_00)

Generates:
  1. Complete alert lists (episodes escalated above tau_esc)
  2. Evidence packages: 3-5s raw MP4, 3-5s annotated MP4, dynamic browser-native GIF, 5-frame filmstrip PNG, detection dump JSON
  3. Verification CSVs (blank labels for human adjudication)
  4. Summary sheet (counts and resource telemetry)
  5. Interactive HTML Visual Adjudication Gallery (gallery.html) opened automatically in browser
"""

import os
import sys
import time
import json
import csv
import glob
import argparse
import hashlib
import psutil
import webbrowser
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.detector_ab_adapters import (
    UltralyticsYOLOAdapter,
    RTDETRv2Adapter,
    LightCoreAdapter,
    StandardDetectionOutput,
)
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch
from road_surveillance.config import load_config
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events.engine import EventEngine

from modules.telemetry_accumulator import TelemetryAccumulator
from modules.episode_consolidator import consolidate_candidates_into_episodes
from modules.scene_escalator import (
    SceneEscalator,
    SceneEscalatorDisposition,
    SceneEscalationResult,
)

ARM_MAP = {
    "Control": ("YOLO26n (Control)", "models/yolo26n.pt", "yolo", {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}),
    "Cand_A": ("UVH-26 YOLO11-S", "models/benchmark_arms/UVH-26-MV-YOLOv11-S.pt", "yolo", {
        0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
        6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
        10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
    }),
    "Cand_B": ("UVH-26 RT-DETRv2-S", "models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth", "rtdetr", {
        0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
        6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
        10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
    }),
    "Cand_C": ("vietnguyen Highway YOLOv8m", "models/benchmark_arms/vietnguyen_best.pt", "yolo", {
        0: "auto_rickshaw", 1: "bus", 2: "car", 3: "truck",
        4: "motorcycle", 5: "truck", 6: "other_vehicle", 7: "truck"
    }),
    "Cand_D": ("G2mus Night YOLOv8s", "models/benchmark_arms/night-vehicle-yolov8s.pt", "yolo", {
        0: "car", 1: "truck", 2: "bus"
    }),
    "Cand_E": ("Nocturnal Light-Core Branch", "internal", "light_core", {
        0: "car", 1: "truck"
    }),
}


def resolve_video_path(video_input: str) -> str:
    """Resolve file path from direct path, clip ID, or dataset archive."""
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
        os.path.join("output", "videos", vid_base),
        os.path.join(".", vid_base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Video '{video_input}' could not be located on disk.")


def load_adapter(arm_key: str):
    name, weights, mtype, unmap = ARM_MAP[arm_key]
    if mtype == "yolo":
        return UltralyticsYOLOAdapter(arm_id=arm_key, model_name=name, weights_path=weights, unified_map=unmap)
    elif mtype == "rtdetr":
        return RTDETRv2Adapter(
            arm_id=arm_key,
            model_name=name,
            config_path="models/benchmark_arms/rtdetr_s_flattened.yaml",
            weights_path=weights,
            unified_map=unmap
        )
    elif mtype == "light_core":
        return LightCoreAdapter(arm_id=arm_key, model_name=name)
    raise ValueError(f"Unknown model type: {mtype}")


def init_pipeline(w: int, h: int, fps: float, conf_th: float, default_calib_cfg: Any):
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id="eval_cam",
        cfg_obj=default_calib_cfg,
        is_calibrated=False,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(w, h))
    lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
    
    tracker = ByteTrackTracker(
        track_activation_threshold=conf_th,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps),
    )
    geometry_engine = GeometryEngine(homography=homography)
    event_engine = EventEngine()
    
    return tracker, geometry_engine, event_engine, lane_manager


def render_filmstrip(frames: List[np.ndarray], labels: List[str], target_width: int = 1280) -> np.ndarray:
    """Generate a horizontal 5-frame filmstrip with text headers."""
    n = len(frames)
    thumb_w = target_width // n
    h_orig, w_orig = frames[0].shape[:2]
    thumb_h = int(h_orig * (thumb_w / w_orig))
    
    strip = np.zeros((thumb_h + 35, target_width, 3), dtype=np.uint8)
    for i, (fr, lbl) in enumerate(zip(frames, labels)):
        x_start = i * thumb_w
        x_end = x_start + thumb_w
        resized = cv2.resize(fr, (thumb_w, thumb_h))
        strip[35:35 + thumb_h, x_start:x_end] = resized
        cv2.rectangle(strip, (x_start, 0), (x_end, 35), (40, 40, 40), -1)
        cv2.rectangle(strip, (x_start, 0), (x_end, thumb_h + 35), (80, 80, 80), 1)
        cv2.putText(strip, lbl, (x_start + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return strip


def generate_html_gallery(output_root: str, clip_id: str):
    """Compile self-contained HTML visual adjudication gallery embedding GIFs and filmstrips."""
    pkg_dir = os.path.join(output_root, "packages")
    sheets_dir = os.path.join(output_root, "verification_sheets")
    out_html = os.path.join(output_root, "gallery.html")

    alert_metadata = {}
    for sheet_file in glob.glob(os.path.join(sheets_dir, "*.csv")):
        sheet_name = os.path.basename(sheet_file)
        parts = sheet_name.replace("verification_", "").replace(".csv", "").split("_tau")
        if len(parts) == 2:
            arm = parts[0]
            tau_str = parts[1]
            try:
                with open(sheet_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        a_idx = row["alert_idx"]
                        c_type = row["esc_type"].replace(" ", "_")
                        key = f"{arm}_tau{tau_str}_{row['clip_id']}_alert{a_idx}_{c_type}"
                        alert_metadata[key] = row
            except Exception:
                pass

    all_gifs = sorted(glob.glob(os.path.join(pkg_dir, "*_annotated.gif")))
    full_gifs = [g for g in all_gifs if "_FULL" in g]
    alert_gifs = [g for g in all_gifs if "_FULL" not in g]

    full_cards = []
    for gif in full_gifs:
        fname = os.path.basename(gif)
        base = fname.replace("_annotated.gif", "")
        ann_mp4 = os.path.join(pkg_dir, base + "_annotated.mp4")
        rel_gif = "packages/" + os.path.basename(gif)
        rel_ann_mp4 = "packages/" + os.path.basename(ann_mp4)

        card = f"""
        <div class="full-card">
            <div class="card-header full-header">
                <h3>{base} &mdash; Complete 30-Second Continuous Recording</h3>
                <span class="badge">Full Video (240 Frames)</span>
            </div>
            <div class="full-media">
                <img src="{rel_gif}" alt="{base}" loading="lazy" />
            </div>
            <div class="card-footer">
                <span>Observe continuous vehicle tracks, bounding boxes, and night visibility across all 30 seconds.</span>
                <div class="file-links">
                    <a href="{rel_ann_mp4}" download>Download Full HD MP4</a>
                </div>
            </div>
        </div>
        """
        full_cards.append(card)

    html_cards = []
    for gif in alert_gifs:
        fname = os.path.basename(gif)
        base = fname.replace("_annotated.gif", "")
        filmstrip = os.path.join(pkg_dir, base + "_filmstrip.png")
        ann_mp4 = os.path.join(pkg_dir, base + "_annotated.mp4")
        raw_mp4 = os.path.join(pkg_dir, base + "_raw.mp4")

        meta = alert_metadata.get(base, {})

        rel_gif = "packages/" + os.path.basename(gif)
        rel_filmstrip = "packages/" + os.path.basename(filmstrip) if os.path.exists(filmstrip) else ""
        rel_ann_mp4 = "packages/" + os.path.basename(ann_mp4)
        rel_raw_mp4 = "packages/" + os.path.basename(raw_mp4)

        card = f"""
        <div class="alert-card">
            <div class="card-header">
                <h3>{base}</h3>
                <span class="badge">Escalation Score: {meta.get('esc_score', 'N/A')}</span>
                <span class="badge type">{meta.get('esc_type', 'Incident')}</span>
                <span class="badge frame">Frame {meta.get('frame', 'N/A')} ({meta.get('timestamp_s', 'N/A')}s)</span>
            </div>
            <div class="media-row">
                <div class="gif-box">
                    <h4>Dynamic Annotated Animation (Browser Playable)</h4>
                    <img src="{rel_gif}" alt="{base}" loading="lazy" />
                </div>
                <div class="filmstrip-box">
                    <h4>5-Frame Key Progression Filmstrip</h4>
                    <a href="{rel_filmstrip}" target="_blank">
                        <img src="{rel_filmstrip}" alt="filmstrip" loading="lazy" />
                    </a>
                </div>
            </div>
            <div class="card-footer">
                <span><b>Contributing Kinematic Channels:</b> {meta.get('contributing_channels', 'None')}</span>
                <div class="file-links">
                    <a href="{rel_ann_mp4}" download>Download Annotated MP4</a> &bull;
                    <a href="{rel_raw_mp4}" download>Download Raw MP4</a> &bull;
                    <a href="{rel_filmstrip}" target="_blank">View Filmstrip HD</a>
                </div>
            </div>
        </div>
        """
        html_cards.append(card)

    full_section_html = ""
    if full_cards:
        full_section_html = f"""
        <div class="full-section">
            <h2>Full Continuous 30-Second Annotated Videos</h2>
            <p style="color:#94a3b8; margin-bottom: 20px;">Watch complete uncut footage to observe how detections and tracking behave continuously from start to finish at each threshold:</p>
            {''.join(full_cards)}
        </div>
        """

    cards_html = ''.join(html_cards) if html_cards else '<p style="padding:20px;">No escalated alerts generated above tau_esc.</p>'
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Road Incident Detector - Verification Gallery ({clip_id})</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
        margin: 0;
        padding: 24px;
        line-height: 1.5;
    }}
    h1 {{ color: #38bdf8; margin-bottom: 6px; }}
    .subtitle {{ color: #94a3b8; margin-bottom: 24px; font-size: 15px; }}
    .stats-banner {{
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
        display: flex;
        gap: 20px;
        border: 1px solid #334155;
    }}
    .stat-item {{ flex: 1; text-align: center; }}
    .stat-value {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
    .stat-label {{ font-size: 13px; color: #94a3b8; }}
    .alert-card {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        margin-bottom: 28px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .card-header {{
        background: #334155;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .card-header h3 {{ margin: 0; font-size: 15px; font-family: monospace; color: #f8fafc; flex-grow: 1; }}
    .badge {{
        background: #0284c7;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }}
    .badge.type {{ background: #d97706; }}
    .badge.frame {{ background: #475569; }}
    .media-row {{
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 18px;
    }}
    @media (min-width: 900px) {{
        .media-row {{ flex-direction: row; }}
    }}
    .gif-box, .filmstrip-box {{
        flex: 1;
        background: #0b1120;
        border-radius: 6px;
        padding: 12px;
        border: 1px solid #1e293b;
    }}
    .gif-box h4, .filmstrip-box h4 {{
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 13px;
        color: #94a3b8;
    }}
    img {{
        width: 100%;
        height: auto;
        border-radius: 4px;
        display: block;
    }}
    .card-footer {{
        background: #182234;
        padding: 12px 18px;
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        border-top: 1px solid #334155;
    }}
    .file-links a {{
        color: #38bdf8;
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
    }}
    .file-links a:hover {{ text-decoration: underline; }}
    .full-section {{
        margin-bottom: 40px;
        background: #111827;
        border: 2px solid #0284c7;
        border-radius: 12px;
        padding: 24px;
    }}
    .full-section h2 {{ color: #38bdf8; margin-top: 0; }}
    .full-card {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 24px;
        overflow: hidden;
    }}
    .full-header {{ background: #0369a1 !important; }}
    .full-media img {{
        width: 100%;
        max-height: 540px;
        object-fit: contain;
        background: #000;
        display: block;
    }}
</style>
</head>
<body>
    <h1>Road Incident Detection - Visual Adjudication Gallery</h1>
    <div class="subtitle">Target Clip: <code>{clip_id}</code> | Showing {len(html_cards)} Escalated Alert Packages</div>
    <div class="stats-banner">
        <div class="stat-item"><div class="stat-value">{len(html_cards)}</div><div class="stat-label">Total Escalated Packages</div></div>
        <div class="stat-item"><div class="stat-value">Interactive Browser Format</div><div class="stat-label">Zero Codec Overhead</div></div>
    </div>
    {full_section_html}
    <h2>Individual Escalated Alert Evidence Packages</h2>
    {cards_html}
</body>
</html>
"""
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive Gallery refreshed at: {out_html}")
    return out_html


def run_threshold_sweep(
    video_path: str,
    output_root: Optional[str] = None,
    arms_to_run: Optional[List[str]] = None,
    conf_thresholds: Optional[List[float]] = None,
    tau_esc: float = 0.50,
    open_browser: bool = False,
):
    resolved_video = resolve_video_path(video_path)
    clip_id = os.path.splitext(os.path.basename(resolved_video))[0]
    
    if output_root is None:
        output_root = "evidence/threshold_sweep_yzNEliTlFGk" if "yzNEliTlFGk" in clip_id else f"evidence/threshold_sweep_{clip_id}"
    
    if arms_to_run is None:
        arms_to_run = ["Cand_B"]
    
    if conf_thresholds is None:
        conf_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
    
    # Sort thresholds ascending
    conf_thresholds = sorted(list(set(conf_thresholds)))
    base_conf = min(conf_thresholds)

    os.makedirs(output_root, exist_ok=True)
    pkg_dir = os.path.join(output_root, "packages")
    sheet_dir = os.path.join(output_root, "verification_sheets")
    os.makedirs(pkg_dir, exist_ok=True)
    os.makedirs(sheet_dir, exist_ok=True)

    with open(resolved_video, "rb") as f_v:
        clip_sha256 = hashlib.sha256(f_v.read()).hexdigest()

    print("=================================================================")
    print("CONFIGURABLE THRESHOLD SWEEP & VISUAL HARNESS")
    print(f"Target Video: {clip_id} (SHA-256: {clip_sha256})")
    print(f"Arms: {arms_to_run}")
    print(f"Confidence Thresholds: {conf_thresholds} | Escalation Threshold: {tau_esc}")
    print(f"Output Root: {output_root}")
    print("=================================================================\n")

    cfg = load_config()
    default_calib = cfg.calibration.get("default")
    proc = psutil.Process()

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

    summary_records = []

    for arm_key in arms_to_run:
        if arm_key not in ARM_MAP:
            print(f"Skipping unknown arm: {arm_key}")
            continue
        
        arm_name = ARM_MAP[arm_key][0]
        print(f"\n>>> Running Arm: {arm_key} ({arm_name})")
        adapter = load_adapter(arm_key)
        
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = adapter.detect(dummy, conf_threshold=base_conf, frame_id=-1)

        # Base inference at base_conf to capture all candidate logits in one pass
        infer_latencies = []
        frame_detections = []
        t0_model = time.perf_counter()
        
        for f_idx, frame in enumerate(raw_frames):
            t0 = time.perf_counter()
            out = adapter.detect(frame, conf_threshold=base_conf, frame_id=f_idx)
            t1 = time.perf_counter()
            infer_latencies.append((t1 - t0) * 1000.0)
            frame_detections.append(out)

        del adapter  # Free model VRAM/RAM
        
        p50_lat = round(float(np.percentile(infer_latencies, 50)), 2)
        p95_lat = round(float(np.percentile(infer_latencies, 95)), 2)
        peak_ram = round(float(proc.memory_info().rss / (1024 * 1024)), 1)

        # Sweep each confidence threshold
        for tau_c in conf_thresholds:
            print(f"  --- Evaluating tau_conf = {tau_c:.2f} ---")
            
            tracker, geom_engine, event_engine, lane_manager = init_pipeline(w, h, fps, conf_th=tau_c, default_calib_cfg=default_calib)
            telemetry_acc = TelemetryAccumulator()
            scene_escalator = SceneEscalator(tau_aftermath=tau_esc)

            all_candidates = []
            frame_annotated = []
            filtered_dets_per_frame = []

            for f_idx, (frame, out_base) in enumerate(zip(raw_frames, frame_detections)):
                mask = out_base.scores >= tau_c
                b_boxes = out_base.boxes[mask]
                b_scores = out_base.scores[mask]
                b_u_ids = out_base.unified_class_ids[mask] if len(out_base.unified_class_ids) > 0 else np.empty((0,), dtype=int)
                b_u_names = [out_base.unified_class_names[i] for i, m in enumerate(mask) if m]
                b_raw_names = [out_base.class_names[i] for i, m in enumerate(mask) if m]

                filtered_dets_per_frame.append(len(b_boxes))

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

                ann_frame = frame.copy()
                for box, score, uname, cid in zip(b_boxes, b_scores, b_u_names, b_u_ids):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ann_frame, f"{uname} {score:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                for tid, tr in active_tracks.items():
                    bx, by, bw, bh = map(int, tr.current_bbox_xywh)
                    cv2.putText(ann_frame, f"ID:{tid}", (bx, by + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(ann_frame, f"Frame: {f_idx:03d} | Time: {timestamp_s:.2f}s | Arm: {arm_key} @ tau={tau_c:.2f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                frame_annotated.append(ann_frame)

            episodes = consolidate_candidates_into_episodes(
                raw_candidates=all_candidates,
                video_id=clip_id,
                cooldown_seconds=10.0,
                max_gap_frames=60,
            )

            escalated_alerts = []
            stream_duration_s = duration_s

            for ep_idx, ep in enumerate(episodes):
                t_anchor = float(ep.peak_candidate["timestamp_s"])
                delta_t_avail = stream_duration_s - t_anchor
                t_end = min(t_anchor + 1.50, stream_duration_s)
                window_history = telemetry_acc.get_window(t_start=t_anchor, t_end=t_end)

                res = scene_escalator.evaluate_candidate(
                    candidate=ep.peak_candidate,
                    track_history=window_history,
                    fps=fps,
                )

                if res.score is not None and res.score >= tau_esc:
                    anchor_f = int(ep.peak_candidate["frame"])
                    active_channels = [f"{k}={v:.2f}" for k, v in res.channels.items() if v > 0.05]
                    channel_str = "; ".join(active_channels) if active_channels else "none"

                    win_start_f = max(0, anchor_f - 15)
                    win_end_f = min(total_frames, anchor_f + 16)
                    contrib_dets = sum(filtered_dets_per_frame[win_start_f:win_end_f])

                    alert_record = {
                        "alert_idx": len(escalated_alerts) + 1,
                        "timestamp_s": round(t_anchor, 3),
                        "frame": anchor_f,
                        "track_ids": ep.peak_candidate["track_ids"],
                        "canonical_type": res.canonical_type,
                        "score": round(float(res.score), 4),
                        "channels": res.channels,
                        "contributing_channels": channel_str,
                        "num_raw_detections": contrib_dets,
                        "win_start_f": win_start_f,
                        "win_end_f": win_end_f,
                    }
                    escalated_alerts.append(alert_record)

            tau_str = f"{tau_c:.2f}".replace(".", "_")

            # Save FULL Continuous Annotated Video & Browser GIF for this threshold
            full_base = f"{arm_key}_tau{tau_str}_{clip_id}_FULL"
            full_mp4 = os.path.join(pkg_dir, f"{full_base}_annotated.mp4")
            writer_full = cv2.VideoWriter(full_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for fr in frame_annotated:
                writer_full.write(fr)
            writer_full.release()

            full_gif = os.path.join(pkg_dir, f"{full_base}_annotated.gif")
            gif_frames = []
            for fr in frame_annotated:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                gh, gw = rgb.shape[:2]
                resized = cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)
                gif_frames.append(Image.fromarray(resized))
            # Mirror to output/annotated_videos
            ann_export_dir = os.path.abspath("output/annotated_videos")
            os.makedirs(ann_export_dir, exist_ok=True)
            import shutil
            shutil.copy2(full_mp4, os.path.join(ann_export_dir, os.path.basename(full_mp4)))
            if os.path.exists(full_gif):
                shutil.copy2(full_gif, os.path.join(ann_export_dir, os.path.basename(full_gif)))

            for alert in escalated_alerts:
                a_idx = alert["alert_idx"]
                a_frame = alert["frame"]
                w_start = alert["win_start_f"]
                w_end = alert["win_end_f"]
                c_type_clean = alert["canonical_type"].replace(" ", "_")

                base_name = f"{arm_key}_tau{tau_str}_{clip_id}_alert{a_idx}_{c_type_clean}"
                
                # Raw MP4
                raw_clip_path = os.path.join(pkg_dir, f"{base_name}_raw.mp4")
                writer_raw = cv2.VideoWriter(raw_clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for f_i in range(w_start, w_end):
                    writer_raw.write(raw_frames[f_i])
                writer_raw.release()

                # Annotated MP4
                ann_clip_path = os.path.join(pkg_dir, f"{base_name}_annotated.mp4")
                writer_ann = cv2.VideoWriter(ann_clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for f_i in range(w_start, w_end):
                    writer_ann.write(frame_annotated[f_i])
                writer_ann.release()
                shutil.copy2(ann_clip_path, os.path.join(ann_export_dir, os.path.basename(ann_clip_path)))

                # Dynamic GIF (Browser compatible)
                gif_path = os.path.join(pkg_dir, f"{base_name}_annotated.gif")
                gif_frames = []
                for f_i in range(w_start, w_end):
                    rgb = cv2.cvtColor(frame_annotated[f_i], cv2.COLOR_BGR2RGB)
                    gh, gw = rgb.shape[:2]
                    resized = cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)
                    gif_frames.append(Image.fromarray(resized))
                if gif_frames:
                    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=int(1000 / fps), loop=0)
                    shutil.copy2(gif_path, os.path.join(ann_export_dir, os.path.basename(gif_path)))

                # 5-frame filmstrip PNG
                f_indices = [
                    w_start,
                    max(w_start, a_frame - 7),
                    a_frame,
                    min(w_end - 1, a_frame + 7),
                    w_end - 1,
                ]
                strip_frames = [frame_annotated[fi] for fi in f_indices]
                strip_labels = [
                    f"F{f_indices[0]} (Pre -15f)",
                    f"F{f_indices[1]} (Approach -7f)",
                    f"F{f_indices[2]} (ANCHOR ALERT)",
                    f"F{f_indices[3]} (Post +7f)",
                    f"F{f_indices[4]} (Aftermath +15f)",
                ]
                filmstrip_img = render_filmstrip(strip_frames, strip_labels)
                filmstrip_path = os.path.join(pkg_dir, f"{base_name}_filmstrip.png")
                cv2.imwrite(filmstrip_path, filmstrip_img)

                # Detections dump JSON
                dump_records = []
                for f_i in range(w_start, w_end):
                    f_out = frame_detections[f_i]
                    m_f = f_out.scores >= tau_c
                    for bx, sc, cl_name, u_name in zip(f_out.boxes[m_f], f_out.scores[m_f], [f_out.class_names[k] for k, m in enumerate(m_f) if m], [f_out.unified_class_names[k] for k, m in enumerate(m_f) if m]):
                        dump_records.append({
                            "frame": f_i,
                            "timestamp_s": round(f_i / fps, 3),
                            "box_xyxy": [round(float(coord), 1) for coord in bx],
                            "score": round(float(sc), 4),
                            "raw_class": cl_name,
                            "unified_class": u_name,
                        })
                dump_json_path = os.path.join(pkg_dir, f"{base_name}_detections.json")
                with open(dump_json_path, "w") as f_json:
                    json.dump(dump_records, f_json, indent=2)

            # Verification sheet CSV
            sheet_csv_path = os.path.join(sheet_dir, f"verification_{arm_key}_tau{tau_str}.csv")
            with open(sheet_csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow([
                    "clip_id", "alert_idx", "timestamp_s", "frame", "esc_score",
                    "esc_type", "contributing_channels", "label", "notes"
                ])
                for alert in escalated_alerts:
                    writer.writerow([
                        clip_id,
                        alert["alert_idx"],
                        alert["timestamp_s"],
                        alert["frame"],
                        alert["score"],
                        alert["canonical_type"],
                        alert["contributing_channels"],
                        "",  # Blank for manual verification (REAL / BENIGN / FALSE)
                        "",
                    ])

            total_dets = sum(filtered_dets_per_frame)
            dets_per_frame = round(total_dets / total_frames, 2)
            alerts_per_hour = round(len(escalated_alerts) / (duration_s / 3600.0), 1)

            summary_records.append({
                "arm": arm_key,
                "model_name": arm_name,
                "tau_conf": tau_c,
                "tau_esc": tau_esc,
                "total_alerts": len(escalated_alerts),
                "alerts_per_hour": alerts_per_hour,
                "alerts_per_clip": len(escalated_alerts),
                "total_detections": total_dets,
                "dets_per_frame": dets_per_frame,
                "p50_latency_ms": p50_lat,
                "p95_latency_ms": p95_lat,
                "peak_ram_mb": peak_ram,
            })

    summary_csv_path = os.path.join(output_root, "summary_sheet.csv")
    new_df = pd.DataFrame(summary_records)
    if os.path.exists(summary_csv_path):
        try:
            existing_df = pd.read_csv(summary_csv_path)
            combined = pd.concat([existing_df, new_df]).drop_duplicates(subset=["arm", "tau_conf", "tau_esc"], keep="last")
            combined.sort_values(by=["arm", "tau_conf"]).to_csv(summary_csv_path, index=False)
        except Exception:
            new_df.to_csv(summary_csv_path, index=False)
    else:
        new_df.to_csv(summary_csv_path, index=False)
    
    # Auto-generate visual gallery
    gallery_path = generate_html_gallery(output_root, clip_id)
    if open_browser:
        try:
            webbrowser.open(os.path.abspath(gallery_path))
        except Exception:
            pass

    print(f"\n========================================================")
    print(f"VERIFICATION HARNESS COMPLETE")
    print(f"Summary Sheet: {summary_csv_path}")
    print(f"Gallery HTML:  {gallery_path}")
    print(f"========================================================")

    return summary_records


def main():
    parser = argparse.ArgumentParser(description="Configurable Road Incident Threshold Verification Harness")
    parser.add_argument("--video", type=str, default=r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\yzNEliTlFGk_00.mp4",
                        help="Path or clip ID of the video to evaluate")
    parser.add_argument("--arms", type=str, default="Cand_B",
                        help="Comma-separated arms to run: Control, Cand_A, Cand_B, Cand_C, Cand_D, or 'all'")
    parser.add_argument("--thresholds", "--conf", type=str, default="0.15,0.20,0.25,0.30,0.35",
                        help="Comma-separated detection confidence thresholds (tau_conf), e.g. '0.15,0.20,0.25,0.30'")
    parser.add_argument("--tau-esc", type=float, default=0.50,
                        help="SceneEscalator escalation threshold (default: 0.50)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: evidence/threshold_sweep_{clip_id})")
    parser.add_argument("--no-browser", action="store_true", default=False,
                        help="Do not automatically open gallery.html in browser")
    parser.add_argument("--list-videos", action="store_true", default=False,
                        help="List available verified night accident videos from metadata-real.csv and exit")

    args = parser.parse_args()

    if args.list_videos:
        meta_candidates = [
            r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\metadata-real.csv",
            "metadata-real.csv",
        ]
        meta_path = next((p for p in meta_candidates if os.path.isfile(p)), None)
        if meta_path:
            import pandas as pd
            df = pd.read_csv(meta_path)
            night = df[(df["day_time"] == "night") & (df["type"] != "FALSE_POSITIVE")]
            print(f"\nFound {len(night)} Verified Night Accident Videos in dataset:\n")
            for _, r in night.head(20).iterrows():
                p = os.path.splitext(os.path.basename(str(r['path'])))[0]
                t = str(r['type'])
                af = str(r['accident_frame'])
                print(f"  --video {p:<22} | Incident: {t:<12} | Impact Frame: {af}")
            print(f"\nTo run on any clip above, pass: --video <clip_id>\n")
        else:
            print("metadata-real.csv not found.")
        return

    if args.arms.lower() == "all":
        arms = ["Control", "Cand_A", "Cand_B", "Cand_C", "Cand_D"]
    else:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    video_inputs = [v.strip() for v in args.video.split(",") if v.strip()]

    for v_path in video_inputs:
        run_threshold_sweep(
            video_path=v_path,
            output_root=args.output_dir,
            arms_to_run=arms,
            conf_thresholds=thresholds,
            tau_esc=args.tau_esc,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
