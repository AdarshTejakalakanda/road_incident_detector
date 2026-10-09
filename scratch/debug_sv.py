import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from ultralytics import YOLO
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO("yolo26n.pt")
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(
    min_cluster_area=250,
    max_cluster_area=3500,
    persistence_threshold=4,
    crop_resolution=320,
    crop_conf_threshold=0.05,
    max_crops_per_frame=1,
)
tracker = RobustSurveillanceTracker()
road_poly = vocab_gate.get_default_roadway_polygon((720, 960))
ingestor = DatasetVideoIngestor(path=vpath)

camera_cfg = CameraConfiguration.from_config_dict("nAXTthLfgtI_00", cfg.calibration.get("default"), is_calibrated=False)
lane_manager = LaneManager.from_config(camera_cfg.lanes, vocab_gate.homography)
geom_engine = GeometryEngine(homography=vocab_gate.homography, lane_manager=lane_manager, is_camera_calibrated=False)
event_engine = EventEngine(config=cfg.events, is_camera_calibrated=False)
event_engine.set_resolution(960, 720)

sv_detector = None
for d in event_engine.detectors:
    if isinstance(d, SingleVehicleAnomalyDetector):
        d.tau_single = 0.50
        d.set_roadway_polygon(road_poly)
        sv_detector = d

last_active = {}
for frame in ingestor.stream_video(vpath):
    f_idx = frame.frame_idx
    if f_idx > 80:
        break
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    yres = yolo_model.predict(source=img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        for bx, bc, bcl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy()):
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": yolo_model.names[int(bcl)].lower(),
            })

    active_boxes = [
        (tr.current_bbox_xywh[0], tr.current_bbox_xywh[1], tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2], tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3])
        for tr in last_active.values()
    ]
    rec_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=raw_dets,
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
        active_track_boxes=active_boxes,
    )
    raw_dets.extend(rec_micro)

    admitted, _ = quality_gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=f_idx,
        timestamp_s=timestamp_s,
        frame_shape=(720, 960),
        homography=vocab_gate.homography,
        roadway_polygon=road_poly,
    )

    tracklets, last_active = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geom_engine.process_frame(tracklets, last_active)
    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="nAXTthLfgtI_00",
        tracks=last_active,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )
    for c in cands:
        print(f"CANDIDATE F{f_idx}: {c.event_type} conf={c.confidence:.2f} tracks={c.track_ids}")

    if 58 <= f_idx <= 63:
        for tid, tr in last_active.items():
            k = kinematics.get(tid)
            print(f"F{f_idx} Track {tid}: label='{tr.class_label}', has_k={k is not None}, age={tr.track_age}, lost={tr.lost_frame_count}")


