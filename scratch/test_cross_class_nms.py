"""Inspect pre-NMS vs post-NMS cross-class suppression on yzNEliTlFGk_00."""

import os
import sys
import torch
import torchvision.ops as ops
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.detector_ab_adapters import RTDETRv2Adapter
from tools.run_threshold_verification_harness import resolve_video_path


def test_cross_class_nms():
    video_path = resolve_video_path("yzNEliTlFGk_00")
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while cap.isOpened() and len(raw_frames) < 100:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    adapter = RTDETRv2Adapter(
        arm_id="Cand_B", model_name="RT-DETR",
        config_path="models/benchmark_arms/rtdetr_s_flattened.yaml",
        weights_path="models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth",
        unified_map={
            0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
            6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
            10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
        }
    )

    print("Analyzing frames 35 to 65 (Collision Window)...")
    stalled_truck_classes = []
    
    for f_idx in range(35, 65):
        frame = raw_frames[f_idx]
        out = adapter.detect(frame, conf_threshold=0.20, frame_id=f_idx)
        
        t_boxes = torch.from_numpy(out.boxes)
        t_scores = torch.from_numpy(out.scores)
        keep = ops.nms(t_boxes, t_scores, iou_threshold=0.50).numpy()

        suppressed_indices = [i for i in range(len(out.boxes)) if i not in keep]
        
        # Check if any large box on left (stalled truck area x1 < 100, y2 > 400) was suppressed
        for i in keep:
            b = out.boxes[i]
            if b[0] < 100 and b[3] > 400 and (b[2] - b[0]) > 400:
                stalled_truck_classes.append((f_idx, out.unified_class_names[i], out.scores[i], out.class_names[i]))

        if len(suppressed_indices) > 0 and f_idx % 5 == 0:
            print(f"\nFrame {f_idx}: {len(out.boxes)} raw boxes -> {len(keep)} post-NMS (suppressed {len(suppressed_indices)})")
            for i in keep:
                print(f"  [KEPT]       {out.unified_class_names[i]:<12} ({out.class_names[i]:<10}) score={out.scores[i]:.3f} box={out.boxes[i].astype(int)}")
            for i in suppressed_indices:
                print(f"  [SUPPRESSED] {out.unified_class_names[i]:<12} ({out.class_names[i]:<10}) score={out.scores[i]:.3f} box={out.boxes[i].astype(int)}")

    print("\n--- Stalled Truck Kept Class Across Frames 35-64 ---")
    import collections
    counts = collections.Counter([c[1] for c in stalled_truck_classes])
    print("Class frequency on stalled truck:", dict(counts))


if __name__ == "__main__":
    test_cross_class_nms()
