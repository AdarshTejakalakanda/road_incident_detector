import sys
import os
sys.path.insert(0, os.path.abspath("."))
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Load metadata
df = pd.read_csv(r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv")

fn_vids = [
    "Z4kg2Ev3vhk_00",
    "nAXTthLfgtI_00",
    "fzWY0vLAXzI_00",
    "VozrseCzhQc_00",
    "kABWTDKgc9U_00",
    "UarP8qU1S-c_00",
    "UarP8qU1S-c_01",
    "82sQ7N-aHOw_00",
]

print("Loading YOLO26n model with full vocabulary and conf=0.05...")
model = YOLO("yolo26n.pt")
names = model.names

results = {}

for vid_id in fn_vids:
    m = df[df["path"].str.contains(vid_id)]
    if len(m) == 0:
        print(f"Video {vid_id} not found in metadata!")
        continue
    row = m.iloc[0]
    gt_frame = int(row["accident_frame"])
    rel_path = row["path"]
    full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive", rel_path)
    if not os.path.exists(full_path):
        # Try real_videos folder directly
        full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", f"{vid_id}.mp4")

    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        continue

    cap = cv2.VideoCapture(full_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n================================================================================")
    print(f"Investigating {vid_id} | GT Frame: {gt_frame} | Total Frames: {total_frames} | FPS: {fps:.1f}")
    print(f"================================================================================")

    # Focus on the collision window: [GT - 35, GT + 35]
    start_f = max(0, gt_frame - 35)
    end_f = min(total_frames - 1, gt_frame + 35)

    window_detections = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    curr_f = start_f

    while curr_f <= end_f:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO with very low floor (0.05) to capture all potential perception signals
        preds = model.predict(source=frame, conf=0.05, imgsz=640, device="cpu", verbose=False)[0]

        if preds.boxes is not None and len(preds.boxes) > 0:
            boxes = preds.boxes.xyxy.cpu().numpy()
            confs = preds.boxes.conf.cpu().numpy()
            clss = preds.boxes.cls.cpu().numpy().astype(int)

            for b, c, cl in zip(boxes, confs, clss):
                cname = names.get(cl, str(cl))
                w = b[2] - b[0]
                h = b[3] - b[1]
                window_detections.append({
                    "frame": curr_f,
                    "rel_f": curr_f - gt_frame,
                    "conf": float(c),
                    "cls_id": cl,
                    "cls_name": cname,
                    "bbox": (float(b[0]), float(b[1]), float(w), float(h)),
                    "area": float(w * h),
                })
        curr_f += 1

    cap.release()

    # Analyze window detections
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
    total_dets = len(window_detections)
    veh_dets = [d for d in window_detections if d["cls_name"] in VEHICLE_CLASSES]
    non_veh_dets = [d for d in window_detections if d["cls_name"] not in VEHICLE_CLASSES]

    veh_high = [d for d in veh_dets if d["conf"] >= 0.25]
    veh_low = [d for d in veh_dets if 0.05 <= d["conf"] < 0.25]

    print(f"Window [f={start_f}..{end_f}] Total detections (conf >= 0.05): {total_dets}")
    print(f"  Vehicle detections (conf >= 0.25): {len(veh_high)}")
    print(f"  Vehicle detections (0.05 <= conf < 0.25): {len(veh_low)}")
    print(f"  Non-vehicle detections: {len(non_veh_dets)}")

    if non_veh_dets:
        nv_classes = {}
        for d in non_veh_dets:
            nv_classes[d["cls_name"]] = nv_classes.get(d["cls_name"], 0) + 1
        print(f"  Non-vehicle classes observed: {nv_classes}")

    # Check temporal continuity around GT [-10..+10]
    near_gt = [d for d in veh_dets if abs(d["rel_f"]) <= 10]
    print(f"  Vehicle detections within +/-10 frames of GT: {len(near_gt)}")
    if near_gt:
        near_gt_frames = sorted(set(d["frame"] for d in near_gt))
        print(f"  Frames with vehicle near GT: {near_gt_frames}")
        for d in near_gt[:10]:
            print(f"    f={d['frame']} (rel={d['rel_f']:+d}) {d['cls_name']:10s} conf={d['conf']:.2f} box=({d['bbox'][0]:.0f},{d['bbox'][1]:.0f},{d['bbox'][2]:.0f},{d['bbox'][3]:.0f})")
    else:
        print("  WARNING: ZERO vehicle detections within +/-10 frames of GT even at conf=0.05!")

    # Classify the primary perception failure mode:
    diagnosis = ""
    if total_dets == 0:
        diagnosis = "GENUINE_DETECTOR_BLINDSPOT (0 detections across entire window)"
    elif len(veh_dets) == 0 and len(non_veh_dets) > 0:
        diagnosis = "CLASS_CONFUSION (Detections present but misclassified as non-vehicle)"
    elif len(near_gt) == 0:
        diagnosis = "TEMPORAL_DROPOUT_AT_IMPACT (Vehicles present before/after, but 0 detections at impact)"
    elif len(veh_high) == 0 and len(veh_low) > 0:
        diagnosis = "LOW_CONFIDENCE_ATTENUATION (Detections present but all conf < 0.25)"
    elif len(veh_high) > 0:
        # Vehicles were detected with conf >= 0.25! Why was it FN?
        # Check if single vehicle or pairwise, and track initialization
        diagnosis = "TRACKER_FAILURE_OR_KINEMATIC_ATTENUATION (Detections exist; track initialization/persistence failed)"
    else:
        diagnosis = "INTERMITTENT_SPARSE_DETECTION"

    print(f"  >> Primary Diagnosis: {diagnosis}")
    results[vid_id] = {
        "gt_frame": gt_frame,
        "total_dets": total_dets,
        "veh_high": len(veh_high),
        "veh_low": len(veh_low),
        "non_veh": len(non_veh_dets),
        "near_gt": len(near_gt),
        "diagnosis": diagnosis,
    }

print("\n================================================================================")
print("Summary of 8 Perception FN Cases:")
for vid, r in results.items():
    print(f"{vid:15s} | GT={r['gt_frame']:3d} | High={r['veh_high']:3d} | Low={r['veh_low']:3d} | NearGT={r['near_gt']:2d} | {r['diagnosis']}")
