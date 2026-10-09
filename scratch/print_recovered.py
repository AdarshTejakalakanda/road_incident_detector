import sys
sys.path.insert(0, ".")
import cv2
from ultralytics import YOLO
from road_surveillance.video.ingestion import DatasetVideoIngestor
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
yolo_model = YOLO("yolo26n.pt")
vocab_gate = VocabularyGate()
motion_gate = MotionROIGate(
    min_cluster_area=250,
    max_cluster_area=3500,
    persistence_threshold=4,
    crop_resolution=320,
    crop_conf_threshold=0.05,
    max_crops_per_frame=1,
)
road_poly = vocab_gate.get_default_roadway_polygon((720, 960))
ingestor = DatasetVideoIngestor(path=vpath)

for frame in ingestor.stream_video(vpath):
    f_idx = frame.frame_idx
    if f_idx > 80:
        break
    img = frame.image
    rec_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=[],
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
    )
    if rec_micro:
        for r in rec_micro:
            bx = [round(x, 1) for x in r["bbox"]]
            w = round(bx[2] - bx[0], 1)
            h = round(bx[3] - bx[1], 1)
            print(f"F{f_idx}: {r['original_class_name']} conf={r['conf']:.2f} bbox={bx} w={w} h={h}")
