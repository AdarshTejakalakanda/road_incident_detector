import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
from ultralytics import YOLO
import supervision as sv

full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", "Z4kg2Ev3vhk_00.mp4")
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(full_path)
fps = cap.get(cv2.CAP_PROP_FPS)

tracker = sv.ByteTrack(
    track_activation_threshold=0.18,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=int(fps),
)
tracker.det_thresh = 0.18

cap.set(cv2.CAP_PROP_POS_FRAMES, 140)
curr_f = 140
tracks_observed = {}

VEHICLE_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "train": "truck",  # Remap roadway train detections to truck
}

while curr_f <= 185:
    ret, frame = cap.read()
    if not ret:
        break
    preds = model.predict(source=frame, conf=0.15, imgsz=640, device="cpu", verbose=False)[0]
    if preds.boxes is not None and len(preds.boxes) > 0:
        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        clss = preds.boxes.cls.cpu().numpy().astype(int)

        keep = []
        mapped_clss = []
        for i, cl in enumerate(clss):
            cname = model.names.get(cl, "")
            if cname in VEHICLE_MAP:
                keep.append(i)
                mapped_clss.append(cl)

        if keep:
            sv_dets = sv.Detections(
                xyxy=boxes[keep],
                confidence=confs[keep],
                class_id=clss[keep],
            )
            tracked = tracker.update_with_detections(sv_dets)
            if len(tracked) > 0 and tracked.tracker_id is not None:
                for tid, b, c, cl in zip(tracked.tracker_id, tracked.xyxy, tracked.confidence, tracked.class_id):
                    tid = int(tid)
                    if tid not in tracks_observed:
                        tracks_observed[tid] = []
                    lbl = VEHICLE_MAP.get(model.names.get(int(cl), ""), "car")
                    tracks_observed[tid].append((curr_f, float(c), lbl, tuple(b)))
    curr_f += 1

cap.release()
print(f"Z4kg2Ev3vhk_00 tracks observed between f=140..185 with train->truck remap: {len(tracks_observed)}")
for tid, frames in tracks_observed.items():
    print(f"  Track {tid:2d}: {frames[0][2]:10s} | {len(frames)} frames [{frames[0][0]}..{frames[-1][0]}], mean_conf={sum(f[1] for f in frames)/len(frames):.2f}, last_box=({frames[-1][3][0]:.0f},{frames[-1][3][1]:.0f},{frames[-1][3][2]-frames[-1][3][0]:.0f},{frames[-1][3][3]-frames[-1][3][1]:.0f})")
