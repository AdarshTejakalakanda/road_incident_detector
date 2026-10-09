import supervision as sv
import numpy as np

tracker = sv.ByteTrack(
    track_activation_threshold=0.18,
    minimum_consecutive_frames=1,
)
tracker.det_thresh = 0.18  # Lower first association threshold!

boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
confs = np.array([0.20], dtype=np.float32)
clss = np.array([2], dtype=int)

for f in range(5):
    dets = sv.Detections(xyxy=boxes, confidence=confs, class_id=clss)
    res = tracker.update_with_detections(dets)
    print(f"frame {f}: len(res)={len(res)}, tracker_id={res.tracker_id}")
