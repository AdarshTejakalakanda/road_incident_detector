"""Pure Object Detection Comparison CLI across Models and Thresholds.

Runs pure object detection (no tracking, no downstream incident rules) across the 5 detector configurations:
  - Arm A: Production YOLO26n (COCO-80)
  - Arm B: UVH-26 YOLO11-S (Bengaluru Safe City)
  - Arm C: UVH-26 RT-DETRv2-S (Transformer Detector)
  - Arm D: vietnguyen Highway YOLOv8m (South Asian Highway)
  - Arm E: G2mus Night YOLOv8s (Night Specialist)

Across customizable confidence thresholds (e.g. 0.10, 0.20, 0.25, 0.30).
Saves all annotated MP4s and browser-playable GIFs directly into:
output/annotated_videos/

Compiles an interactive side-by-side comparison matrix (detector_comparison_matrix.html).
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_annotated_detector import get_adapter, resolve_video_path

# Vivid unified palette for clear visual review
CLASS_COLORS = {
    "truck": (0, 220, 0),        # Vivid Green
    "car": (255, 165, 0),        # Vivid Cyan/Blue (BGR)
    "van": (210, 105, 30),       # Chocolate/Violet
    "bus": (0, 140, 255),        # Deep Orange
    "motorcycle": (0, 255, 255), # Yellow
    "auto_rickshaw": (255, 0, 255), # Magenta
    "bicycle": (180, 105, 255),  # Hot Pink
    "other_vehicle": (200, 200, 200), # Light Gray
}
DEFAULT_COLOR = (0, 255, 128)

ARM_KEYS = {
    "A": ("Arm_A_YOLO26n_Baseline", "Production Baseline (YOLO26n COCO)"),
    "B": ("Arm_B_UVH_YOLO11s", "UVH-26 Reference (YOLO11-S 14-Class)"),
    "C": ("Arm_C_RT_DETRv2_S", "UVH-26 Transformer (RT-DETRv2-S)"),
    "D": ("Arm_D_Highway_v8m", "Highway Traffic (vietnguyen YOLOv8m)"),
    "E": ("Arm_E_Night_v8s", "Night Specialist (G2mus YOLOv8s)"),
    "F": ("Arm_F_BDD100K_YOLO11s", "BDD100K Night Traffic (YOLO11-S)"),
}


def draw_detection_frame(frame: np.ndarray, boxes, scores, unified_names, arm_label: str, threshold: float, f_idx: int, total_f: int, timestamp_s: float) -> np.ndarray:
    """Renders high-visibility bounding boxes, class tags, and diagnostic HUD."""
    ann = frame.copy()
    h, w = ann.shape[:2]

    # Draw Bounding Boxes
    for box, score, uname in zip(boxes, scores, unified_names):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(uname, DEFAULT_COLOR)
        cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)

        # Label background pill
        label_text = f"{uname} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y_text = max(th + 4, y1 - 4)
        cv2.rectangle(ann, (x1, y_text - th - 3), (x1 + tw + 4, y_text + baseline), (20, 20, 20), -1)
        cv2.rectangle(ann, (x1, y_text - th - 3), (x1 + tw + 4, y_text + baseline), color, 1)
        cv2.putText(ann, label_text, (x1 + 2, y_text - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Top HUD Bar
    hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.rectangle(hud_bg, (0, 0), (w, 40), (15, 23, 42), -1)
    cv2.line(hud_bg, (0, 39), (w, 39), (51, 65, 85), 1)

    hud_left = f"{arm_label} | tau={threshold:.2f}"
    hud_right = f"Frame: {f_idx:03d}/{total_f} ({timestamp_s:.2f}s) | Objects: {len(boxes)}"
    
    cv2.putText(hud_bg, hud_left, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(hud_bg, hud_right, (w - rw - 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (241, 245, 249), 1, cv2.LINE_AA)

    combined = np.vstack([hud_bg, ann])
    return combined


def build_matrix_dashboard(out_dir: str, clip_id: str, generated_items: List[Dict[str, Any]]) -> str:
    """Compiles an interactive comparison matrix HTML page."""
    html_path = os.path.join(out_dir, "detector_comparison_matrix.html")

    # Group by arm and threshold
    cards_html = []
    unique_taus = sorted(list(set(item["tau"] for item in generated_items)))
    tau_buttons = "\n        ".join([f'<button class="filter-btn" onclick="filterTau(\'{t:.2f}\')">tau = {t:.2f}</button>' for t in unique_taus])

    for item in generated_items:
        arm_slug = item["arm_slug"]
        arm_name = item["arm_name"]
        tau = item["tau"]
        mp4_name = os.path.basename(item["mp4_path"])
        gif_name = os.path.basename(item["gif_path"])
        dets_avg = item["avg_dets"]

        card = f"""
        <div class="model-card" data-arm="{item['arm_code']}" data-tau="{tau:.2f}">
            <div class="card-header">
                <h3>{arm_name}</h3>
                <span class="badge tau">tau = {tau:.2f}</span>
                <span class="badge dets">Avg {dets_avg:.1f} objs/frame</span>
            </div>
            <div class="media-container">
                <img src="{gif_name}" alt="{arm_slug}" loading="lazy" />
            </div>
            <div class="card-footer">
                <span>File: <code>{mp4_name}</code></span>
                <a href="{mp4_name}" download class="btn-dl">Download Full HD MP4</a>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detector A/B/C/D/E Object Detection Comparison Matrix</title>
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
    .subtitle {{ color: #94a3b8; margin-bottom: 20px; font-size: 15px; }}
    .filter-bar {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 24px;
        display: flex;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
    }}
    .filter-label {{ font-size: 14px; font-weight: 600; color: #38bdf8; }}
    .filter-btn {{
        background: #334155;
        color: #f8fafc;
        border: 1px solid #475569;
        padding: 6px 14px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        transition: all 0.2s;
    }}
    .filter-btn.active {{
        background: #0284c7;
        border-color: #38bdf8;
        font-weight: bold;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
        gap: 24px;
    }}
    .model-card {{
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
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }}
    .badge.tau {{ background: #0284c7; color: white; }}
    .badge.dets {{ background: #334155; color: #94a3b8; }}
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
        padding: 6px 12px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 12px;
        font-weight: 600;
    }}
    .btn-dl:hover {{ background: #0369a1; }}
</style>
</head>
<body>
    <h1>Detector Models A/B/C/D/E Object Detection Comparison</h1>
    <div class="subtitle">Target: <code>{clip_id}.mp4</code> | Pure Object Detection (Bounding Boxes & Scores Only)</div>

    <div class="filter-bar">
        <span class="filter-label">Filter by Threshold:</span>
        <button class="filter-btn active" onclick="filterTau('all')">Show All</button>
        {tau_buttons}
    </div>

    <div class="grid" id="cardGrid">
        {''.join(cards_html)}
    </div>

    <script>
    function filterTau(val) {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        event.target.classList.add('active');
        document.querySelectorAll('.model-card').forEach(card => {{
            if (val === 'all' || card.getAttribute('data-tau') === val) {{
                card.style.display = 'block';
            }} else {{
                card.style.display = 'none';
            }}
        }});
    }}
    </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\nInteractive Comparison Matrix HTML: {html_path}")
    return html_path


def run_detection_comparison(
    video_input: str = "yzNEliTlFGk_00",
    arms: List[str] = ["A", "B", "C", "D", "E"],
    thresholds: List[float] = [0.10, 0.20, 0.25, 0.30],
    output_dir: str = r"C:\Users\Adarsh Teja Kalakand\Desktop\road_incidents_detector\output\annotated_videos",
):
    resolved_video = resolve_video_path(video_input)
    clip_id = os.path.splitext(os.path.basename(resolved_video))[0]
    os.makedirs(output_dir, exist_ok=True)

    print("=================================================================")
    print("PURE OBJECT DETECTION COMPARISON ACROSS MODELS & THRESHOLDS")
    print(f"Target Video: {clip_id}")
    print(f"Detector Arms: {arms}")
    print(f"Thresholds: {thresholds}")
    print(f"Output Directory: {output_dir}")
    print("=================================================================\n")

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
    print(f"Preloaded {total_frames} frames ({w}x{h} @ {fps:.2f} FPS, {duration_s:.2f}s)\n")

    thresholds = sorted(list(set(thresholds)))
    base_conf = min(thresholds)

    generated_items = []

    for arm in arms:
        arm = arm.upper()
        if arm not in ARM_KEYS:
            print(f"Skipping unknown arm: {arm}")
            continue

        arm_slug, arm_name = ARM_KEYS[arm]
        print(f"\n>>> Running Detector [{arm}]: {arm_name}")
        adapter = get_adapter(arm)

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = adapter.detect(dummy, conf_threshold=base_conf, frame_id=-1)

        # Single-pass inference at lowest threshold
        print(f"Extracting detection boxes at base_conf={base_conf:.2f}...")
        t0 = time.perf_counter()
        raw_detections = []
        for f_idx, frame in enumerate(raw_frames):
            out = adapter.detect(frame, conf_threshold=base_conf, frame_id=f_idx)
            raw_detections.append(out)
        t_infer = time.perf_counter() - t0
        print(f"Inference completed in {t_infer:.1f}s ({total_frames/t_infer:.1f} FPS)")
        del adapter

        # Render each threshold video
        for tau in thresholds:
            tau_str = f"{tau:.2f}".replace(".", "_")
            out_mp4_name = f"{arm_slug}_tau{tau_str}_{clip_id}_annotated.mp4"
            out_gif_name = f"{arm_slug}_tau{tau_str}_{clip_id}_annotated.gif"
            out_mp4_path = os.path.join(output_dir, out_mp4_name)
            out_gif_path = os.path.join(output_dir, out_gif_name)

            print(f"  Rendering tau={tau:.2f} -> {out_mp4_name}...")
            
            # 40px HUD header added
            writer = cv2.VideoWriter(out_mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 40))
            gif_frames = []
            total_dets_tau = 0

            for f_idx, (frame, out_base) in enumerate(zip(raw_frames, raw_detections)):
                mask = out_base.scores >= tau
                b_boxes = out_base.boxes[mask]
                b_scores = out_base.scores[mask]
                b_u_names = [out_base.unified_class_names[i] for i, m in enumerate(mask) if m]
                total_dets_tau += len(b_boxes)

                ann_frame = draw_detection_frame(
                    frame=frame,
                    boxes=b_boxes,
                    scores=b_scores,
                    unified_names=b_u_names,
                    arm_label=f"{arm_slug}",
                    threshold=tau,
                    f_idx=f_idx,
                    total_f=total_frames,
                    timestamp_s=f_idx / fps,
                )
                writer.write(ann_frame)

                # Resize for responsive browser GIF
                rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                gh, gw = rgb.shape[:2]
                gif_frames.append(Image.fromarray(cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)))

            writer.release()
            if gif_frames:
                gif_frames[0].save(
                    out_gif_path,
                    save_all=True,
                    append_images=gif_frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )

            avg_dets = total_dets_tau / total_frames
            print(f"    Total Objects: {total_dets_tau} ({avg_dets:.2f}/frame) | Saved MP4 & GIF")

            generated_items.append({
                "arm_code": arm,
                "arm_slug": arm_slug,
                "arm_name": arm_name,
                "tau": tau,
                "mp4_path": out_mp4_path,
                "gif_path": out_gif_path,
                "total_dets": total_dets_tau,
                "avg_dets": avg_dets,
            })

    # Build Interactive HTML Matrix Dashboard
    matrix_html = build_matrix_dashboard(output_dir, clip_id, generated_items)
    return matrix_html, generated_items


def main():
    parser = argparse.ArgumentParser(description="Pure Object Detection Comparison across Models & Thresholds")
    parser.add_argument("--video", type=str, default="yzNEliTlFGk_00",
                        help="Video filename, ID, or path (default: yzNEliTlFGk_00)")
    parser.add_argument("--arms", type=str, default="A,B,C,D,E",
                        help="Comma-separated arms to run: A, B, C, D, E or 'all'")
    parser.add_argument("--thresholds", "--conf", type=str, default="0.10,0.20,0.25,0.30",
                        help="Comma-separated thresholds, e.g. '0.10,0.20,0.25,0.30'")
    parser.add_argument("--output-dir", type=str,
                        default=r"C:\Users\Adarsh Teja Kalakand\Desktop\road_incidents_detector\output\annotated_videos",
                        help="Output directory for annotated videos")

    args = parser.parse_args()

    if args.arms.lower() == "all":
        arms = ["A", "B", "C", "D", "E"]
    else:
        arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]

    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]

    matrix_html, items = run_detection_comparison(
        video_input=args.video,
        arms=arms,
        thresholds=thresholds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
