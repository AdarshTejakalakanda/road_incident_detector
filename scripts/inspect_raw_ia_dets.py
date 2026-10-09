import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from road_surveillance.detection import YOLO26Detector
from road_surveillance.video import DatasetVideoIngestor

detector = YOLO26Detector("yolo26n.pt", conf_threshold=0.10, device="cpu")
v_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", "iA_u73BG4ZM_00.mp4")

ingestor = DatasetVideoIngestor(v_path)
for f in ingestor.stream_video(v_path):
    if 105 <= f.frame_idx <= 125:
        batch = detector.detect(f)
        dets = [(batch.class_labels[i], round(float(batch.confidences[i]), 2), [round(float(x), 1) for x in batch.boxes[i]]) for i in range(len(batch.boxes))]
        print(f"f={f.frame_idx:3d}: {dets}")
