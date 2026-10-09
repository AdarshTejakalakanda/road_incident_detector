"""Deep inspection of tracks, detections, and candidate conditions around GT window."""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from road_surveillance.detection.yolo_detector import YOLO26Detector
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.video.ingestion import IngestedFrame

def inspect(vid_id, gt_frame, imgsz=640):
    video_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", f"{vid_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n{'='*70}\nDeep inspecting {vid_id} (GT={gt_frame}, FPS={fps:.1f}, imgsz={imgsz})\n{'='*70}")

    detector = YOLO26Detector("yolo26n.pt", conf_threshold=0.10, input_resolution=imgsz, device="cpu", enable_adaptive_contrast=True)
    tracker = ByteTrackTracker(track_activation_threshold=0.15, frame_rate=int(fps))
    h = PlanarHomography()
    geom = GeometryEngine(h)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pts_ns = int(frame_idx * (1e9 / fps))
        ingested = IngestedFrame(
            camera_id="cam_01",
            video_id=vid_id,
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            fps=fps,
            image=frame_bgr,
        )

        batch = detector.detect(ingested)
        tl, tracks = tracker.update(batch)
        kin, conv = geom.process_frame(tl, tracks)

        if abs(frame_idx - gt_frame) <= 15:
            # Print detailed tracking & kinematic state
            print(f"--- Frame {frame_idx} (rel {frame_idx - gt_frame:+d}f) ---")
            print(f"  Detections: {len(batch.boxes)} | Confidences: {[round(float(c), 2) for c in batch.confidences]} | Labels: {batch.class_labels}")
            print(f"  Active Tracks ({len(tracks)}):")
            for tid, tr in tracks.items():
                k = kin.get(tid)
                spd = f"{k.speed_kmh:.1f} km/h" if k else "N/A"
                acc = f"{k.acceleration_mag:.1f} m/s2" if k else "N/A"
                print(f"    Track {tid}: label={tr.class_label} (eff={tr.get_effective_label()}) age={tr.track_age} status={tr.status} spd={spd} acc={acc} bbox={[round(x, 1) for x in tr.current_bbox_xywh]}")

        frame_idx += 1
    cap.release()

if __name__ == "__main__":
    inspect("82sQ7N-aHOw_00", 15, imgsz=640)
    inspect("UarP8qU1S-c_00", 130, imgsz=640)
