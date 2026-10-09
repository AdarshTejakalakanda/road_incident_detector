import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from road_surveillance.detection import YOLO26Detector
from road_surveillance.video import DatasetVideoIngestor

detector = YOLO26Detector("yolo26n.pt", conf_threshold=0.10, device="cpu")

for vid_name in ["nAXTthLfgtI_00.mp4", "VozrseCzhQc_00.mp4"]:
    v_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", vid_name)
    print(f"\nTesting {vid_name} exists: {os.path.exists(v_path)}")
    ingestor = DatasetVideoIngestor(v_path)
    det_counts = []
    sample_dets = []
    for f in ingestor.stream_video(v_path):
        batch = detector.detect(f)
        det_counts.append(len(batch.boxes))
        if len(batch.boxes) > 0 and len(sample_dets) < 5:
            sample_dets.append((f.frame_idx, list(zip(batch.class_labels, [round(float(c), 2) for c in batch.confidences]))))
    print(f"Total frames: {len(det_counts)}, frames with dets (conf>=0.10): {sum(1 for c in det_counts if c > 0)}")
    print(f"Sample detections: {sample_dets}")
