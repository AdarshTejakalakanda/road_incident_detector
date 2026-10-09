"""Investigate why supervision ByteTrack lost tracks in UarP8qU1S-c_00."""

import os
import sys
import cv2
import numpy as np
import supervision as sv

sys.path.insert(0, os.path.abspath("."))
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\UarP8qU1S-c_00.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

tracker = sv.ByteTrack(
    track_activation_threshold=0.15,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=int(fps),
)
tracker.det_thresh = 0.15

f = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if 115 <= f <= 135:
        preds = model.predict(source=frame, conf=0.10, imgsz=640, device="cpu", verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        clss = preds.boxes.cls.cpu().numpy().astype(int)

        sv_dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=clss)
        res = tracker.update_with_detections(sv_dets)

        print(f"Frame {f:3d} (rel {f-130:+2d}): Raw Dets={len(boxes)} {[round(float(c), 2) for c in confs]}")
        print(f"   Detections xyxy: {[list(np.round(b, 1)) for b in boxes]}")
        tids = list(res.tracker_id) if res.tracker_id is not None else []
        print(f"   Tracked IDs: {tids}")
        if len(tids) > 0:
            print(f"   Tracked xyxy: {[list(np.round(b, 1)) for b in res.xyxy]}")
    else:
        # Just update tracker so state persists
        preds = model.predict(source=frame, conf=0.10, imgsz=640, device="cpu", verbose=False)[0]
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        clss = preds.boxes.cls.cpu().numpy().astype(int)
        sv_dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=clss)
        tracker.update_with_detections(sv_dets)

    f += 1

cap.release()
