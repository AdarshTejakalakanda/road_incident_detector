import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np

from road_surveillance.video.ingestion import IngestedFrame
from road_surveillance.detection.yolo_detector import YOLO26Detector
from road_surveillance.tracking.tracker import ByteTrackTracker

full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", "82sQ7N-aHOw_00.mp4")
cap = cv2.VideoCapture(full_path)
fps = cap.get(cv2.CAP_PROP_FPS)

detector = YOLO26Detector(conf_threshold=0.18)
tracker = ByteTrackTracker(track_activation_threshold=0.18, frame_rate=int(fps))

print("Running YOLO26 + ByteTrack on 82sQ7N-aHOw_00 from f=0 to f=30:")
tracks_active = {}
for f in range(30):
    ret, frame = cap.read()
    if not ret:
        break
    pts_ns = int(f * (1e9 / fps))
    ingested = IngestedFrame(
        camera_id="cam_0",
        video_id="82sQ7N-aHOw_00",
        frame_idx=f,
        pts_ns=pts_ns,
        fps=fps,
        image=frame,
    )
    batch = detector.detect(ingested)
    tracklets, tracks = tracker.update(batch)
    for tlet in tracklets:
        if tlet.track_id not in tracks_active:
            tracks_active[tlet.track_id] = {"frames": [], "label": tlet.class_label}
        tracks_active[tlet.track_id]["frames"].append(f)

cap.release()
print(f"Tracks created: {len(tracks_active)}")
for tid, d in tracks_active.items():
    print(f"  Track {tid:2d}: {d['label']:10s} across {len(d['frames'])} frames [{min(d['frames'])}..{max(d['frames'])}]")
