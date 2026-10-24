"""Batch Annotated Video Generator for Path C Night Survey Clips.

Generates full-length annotated MP4s and browser-playable GIFs for:
  1. unS0-TLF1ao_00 (Urban T-Bone, 149 frames)
  2. B_nBw3PcHXg_00 (Illuminated Highway Rear-End, 476 frames)
  3. Z4kg2Ev3vhk_00 (Arterial Highway Rear-End, 425 frames)
  4. FR3eYZzgeJY_00 (City Junction T-Bone, 366 frames)

Saves outputs to output/annotated_videos/ and compiles path_c_verification_gallery.html.
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_annotated_detector import resolve_video_path
from ultralytics import YOLO

# Palette
COLORS = {
    "car": (255, 165, 0),       # Vivid Blue/Cyan (BGR)
    "truck": (0, 220, 0),       # Vivid Green
    "lcv": (0, 200, 100),       # Spring Green
    "bus": (0, 140, 255),       # Deep Orange
    "motorcycle": (0, 255, 255),# Yellow
    "auto_rickshaw": (255, 0, 255),
    "other_vehicle": (200, 200, 200),
}
DEFAULT_COLOR = (0, 255, 128)

TARGET_CLIPS = [
    {
        "clip_id": "unS0-TLF1ao_00",
        "name": "unS0-TLF1ao_00",
        "impact_frame": 64,
        "type": "T-Bone Crash",
        "desc": "Urban lighted intersection; cross-traffic collision.",
        "model_path": "models/benchmark_arms/vietnguyen_best.pt",
        "model_label": "Highway YOLOv8m (Arm D)",
    },
    {
        "clip_id": "B_nBw3PcHXg_00",
        "name": "B_nBw3PcHXg_00",
        "impact_frame": 75,
        "type": "Rear-End Crash",
        "desc": "Overhead illuminated multi-lane highway; high-speed rear-end collision.",
        "model_path": "models/benchmark_arms/vietnguyen_best.pt",
        "model_label": "Highway YOLOv8m (Arm D)",
    },
    {
        "clip_id": "Z4kg2Ev3vhk_00",
        "name": "Z4kg2Ev3vhk_00",
        "impact_frame": 170,
        "type": "Rear-End Crash",
        "desc": "Arterial highway night traffic; stopped vehicle struck from behind.",
        "model_path": "models/benchmark_arms/vietnguyen_best.pt",
        "model_label": "Highway YOLOv8m (Arm D)",
    },
    {
        "clip_id": "FR3eYZzgeJY_00",
        "name": "FR3eYZzgeJY_00",
        "impact_frame": 38,
        "type": "T-Bone Crash",
        "desc": "City junction under streetlights; turning vehicle T-bone.",
        "model_path": "models/yolo26n.pt",
        "model_label": "YOLO26n Baseline (Arm A)",
    },
]


def render_clip(info: Dict, output_dir: str, tau: float = 0.25) -> Tuple[str, str, int, float]:
    clip_id = info["clip_id"]
    video_path = resolve_video_path(clip_id)
    impact_f = info["impact_frame"]
    model_label = info["model_label"]
    
    out_mp4 = os.path.join(output_dir, f"{clip_id}_tau{int(tau*100)}_annotated.mp4")
    out_gif = os.path.join(output_dir, f"{clip_id}_tau{int(tau*100)}_annotated.gif")
    
    print(f"\n>>> Rendering: {clip_id} ({info['type']})")
    print(f"    Model: {model_label} | tau = {tau:.2f}")
    
    model = YOLO(info["model_path"])
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 40))
    gif_frames = []
    
    f_idx = 0
    total_objects = 0
    t0 = time.perf_counter()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        ts_s = f_idx / fps
        res = model.predict(frame, conf=tau, imgsz=640, verbose=False)[0]
        
        ann = frame.copy()
        n_dets = len(res.boxes)
        total_objects += n_dets
        
        for b in res.boxes:
            cname = model.names[int(b.cls[0])]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            color = COLORS.get(cname.lower(), DEFAULT_COLOR)
            
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cname} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y_text = max(th + 4, y1 - 4)
            cv2.rectangle(ann, (x1, y_text - th - 3), (x1 + tw + 4, y_text + baseline), (20, 20, 20), -1)
            cv2.rectangle(ann, (x1, y_text - th - 3), (x1 + tw + 4, y_text + baseline), color, 1)
            cv2.putText(ann, lbl, (x1 + 2, y_text - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            
        # Top HUD Bar
        hud = np.zeros((40, w, 3), dtype=np.uint8)
        # Change HUD color to alert red near impact frame (impact +- 15 frames)
        is_impact_window = abs(f_idx - impact_f) <= 15
        hud_bg_color = (15, 23, 70) if is_impact_window else (15, 23, 42)
        cv2.rectangle(hud, (0, 0), (w, 40), hud_bg_color, -1)
        cv2.line(hud, (0, 39), (w, 39), (239, 68, 68) if is_impact_window else (51, 65, 85), 2 if is_impact_window else 1)
        
        status_tag = f"CRASH EVENT (Impact @ f{impact_f})" if is_impact_window else f"{info['type']} (Impact @ f{impact_f})"
        hud_left = f"{clip_id} | {status_tag} | {model_label}"
        hud_right = f"Frame: {f_idx:03d}/{total_frames} ({ts_s:.2f}s) | Objects: {n_dets}"
        
        cv2.putText(hud, hud_left, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (239, 68, 68) if is_impact_window else (56, 189, 248), 1, cv2.LINE_AA)
        (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(hud, hud_right, (w - rw - 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (241, 245, 249), 1, cv2.LINE_AA)
        
        combined = np.vstack([hud, ann])
        writer.write(combined)
        
        # Capture frames for browser GIF (sample every 3rd frame to keep size lean)
        if f_idx % 3 == 0:
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            gh, gw = rgb.shape[:2]
            # Scale down for browser performance
            target_w = 640
            scale = target_w / float(gw)
            target_h = int(gh * scale)
            gif_frames.append(Image.fromarray(cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)))
            
        f_idx += 1
        
    cap.release()
    writer.release()
    t_elapsed = time.perf_counter() - t0
    
    print(f"    Rendered {total_frames} frames in {t_elapsed:.1f}s ({total_frames/t_elapsed:.1f} FPS)")
    print(f"    Avg Objects/Frame: {total_objects/total_frames:.2f} | Total: {total_objects}")
    
    if gif_frames:
        print(f"    Encoding GIF ({len(gif_frames)} frames)...")
        gif_frames[0].save(
            out_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(3000 / fps),
            loop=0,
        )
    return out_mp4, out_gif, total_objects, total_objects / max(1, total_frames)


def build_verification_gallery(results: List[Dict], out_dir: str):
    html_path = os.path.join(out_dir, "path_c_verification_gallery.html")
    
    cards_html = []
    for r in results:
        mp4_name = os.path.basename(r["mp4_path"])
        gif_name = os.path.basename(r["gif_path"])
        card = f"""
        <div class="video-card">
            <div class="card-header">
                <h3>{r['clip_id']} &mdash; <span style="color: #38bdf8;">{r['type']}</span></h3>
                <span class="badge impact">Impact Frame: {r['impact_frame']}</span>
                <span class="badge model">{r['model_label']}</span>
                <span class="badge stats">Avg {r['avg_dets']:.1f} objs/frame</span>
            </div>
            <div class="card-desc">
                {r['desc']}
            </div>
            <div class="media-container">
                <img src="{gif_name}" alt="{r['clip_id']}" loading="lazy" />
            </div>
            <div class="card-footer">
                <span>MP4 File: <code>{mp4_name}</code></span>
                <a href="{mp4_name}" download class="btn-dl">Download Full-Res MP4</a>
            </div>
        </div>
        """
        cards_html.append(card)
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Path C Night Survey &mdash; Full Annotated Videos for Manual Verification</title>
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
    .banner {{
        background: #1e293b;
        border-left: 4px solid #10b981;
        padding: 14px 20px;
        border-radius: 6px;
        margin-bottom: 24px;
        font-size: 14px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
        gap: 24px;
    }}
    .video-card {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .card-header {{
        background: #182234;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        border-bottom: 1px solid #334155;
    }}
    .card-header h3 {{ margin: 0; font-size: 16px; color: #f8fafc; flex-grow: 1; }}
    .badge {{
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }}
    .badge.impact {{ background: #dc2626; color: white; }}
    .badge.model {{ background: #0284c7; color: white; }}
    .badge.stats {{ background: #334155; color: #cbd5e1; }}
    .card-desc {{
        padding: 10px 18px;
        background: #151f30;
        font-size: 13px;
        color: #94a3b8;
        border-bottom: 1px solid #293548;
    }}
    .media-container img {{
        width: 100%;
        height: auto;
        display: block;
        background: #000;
    }}
    .card-footer {{
        background: #182234;
        padding: 12px 18px;
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-top: 1px solid #334155;
    }}
    .btn-dl {{
        background: #0284c7;
        color: white;
        padding: 6px 14px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 12px;
        font-weight: 600;
    }}
    .btn-dl:hover {{ background: #0369a1; }}
</style>
</head>
<body>
    <h1>Path C Night Survey: Full Annotated Videos</h1>
    <div class="subtitle">Manual Verification Gallery Across Diverse Night Crash Scenes (&tau; = 0.25)</div>

    <div class="banner">
        <strong>Manual Adjudication Guide:</strong> Watch the browser animations below or download the full-resolution MP4s. Observe how the detector locks onto colliding vehicles at the red HUD impact callout window, and evaluate track stability across both highway and intersection night scenes.
    </div>

    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nVerification Gallery compiled at: {html_path}")
    return html_path


def main():
    out_dir = r"output\annotated_videos"
    os.makedirs(out_dir, exist_ok=True)
    
    print("================================================================")
    print("PATH C BATCH ANNOTATED VIDEO GENERATOR")
    print(f"Output Directory: {out_dir}")
    print(f"Number of Clips: {len(TARGET_CLIPS)}")
    print("================================================================")
    
    results = []
    for info in TARGET_CLIPS:
        mp4_path, gif_path, total_dets, avg_dets = render_clip(info, out_dir, tau=0.25)
        results.append({
            **info,
            "mp4_path": mp4_path,
            "gif_path": gif_path,
            "total_dets": total_dets,
            "avg_dets": avg_dets,
        })
        
    gallery_path = build_verification_gallery(results, out_dir)
    return gallery_path


if __name__ == "__main__":
    main()
