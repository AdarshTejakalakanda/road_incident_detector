import os, sys, json
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from road_surveillance.config import load_config
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from ultralytics import YOLO

video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(crop_resolution=320, crop_conf_threshold=0.05, max_crops_per_frame=1, min_crop_interval=2)

cap_meta = cv2.VideoCapture(video_path)
vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
cap_meta.release()

raw_calib = cfg.calibration.get("Z4kg2Ev3vhk_00") or cfg.calibration.get("default")
camera_cfg = CameraConfiguration.from_config_dict("Z4kg2Ev3vhk_00", raw_calib, is_calibrated=False)
homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
geometry_engine = GeometryEngine(homography, lane_manager, False)
event_engine = EventEngine(cfg.events, False)
event_engine.set_resolution(vid_w, vid_h)
tracker = RobustSurveillanceTracker()

ingestor = DatasetVideoIngestor(video_path)

print("Streaming through F210...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 210:
        break
    img = frame.image
    proc_img, _ = preprocessor.process(img)
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    yres = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_yolo_dets = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        b_xyxy = yres.boxes.xyxy.cpu().numpy()
        b_conf = yres.boxes.conf.cpu().numpy()
        b_cls = yres.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            raw_yolo_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": yolo_model.names[int(bcl)].lower()
            })

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    recovered_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=[{"bbox": d["bbox"], "conf": d["conf"], "class_id": 2, "class_name": d["class_name"]} for d in raw_yolo_dets],
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
    )
    all_dets = list(raw_yolo_dets)
    for rm in recovered_micro:
        all_dets.append({
            "bbox": [float(x) for x in rm.get("bbox", [])],
            "conf": float(rm.get("conf", 0.0)),
            "class_id": 2,
            "class_name": rm.get("class_name", "recovered"),
        })

    admitted_recs, _ = quality_gate.process_and_admit(
        all_dets, f_idx, timestamp_s, (vid_h, vid_w), homography, roadway_polygon=road_poly
    )
    tracklets, track_dict = tracker.update(admitted_recs, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geometry_engine.process_frame(tracklets, track_dict)
    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="Z4kg2Ev3vhk_00",
        tracks=track_dict,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )

    if 170 <= f_idx <= 210:
        if cands:
            for c in cands:
                print(f"F{f_idx} CANDIDATE: {c.event_type} tracks={c.track_ids} conf={c.confidence:.2f} desc={c.description}")
        t10 = track_dict.get(10)
        t22 = track_dict.get(22)
        if t10 and t22:
            col_det = [d for d in event_engine.detectors if hasattr(d, "suppressed_events")][0]
            iou = col_det._compute_bbox_iou(t10.current_bbox_xywh, t22.current_bbox_xywh)
            k10 = kinematics.get(10)
            k22 = kinematics.get(22)
            c10 = (t10.current_bbox_xywh[0] + t10.current_bbox_xywh[2]/2, t10.current_bbox_xywh[1] + t10.current_bbox_xywh[3]/2)
            c22 = (t22.current_bbox_xywh[0] + t22.current_bbox_xywh[2]/2, t22.current_bbox_xywh[1] + t22.current_bbox_xywh[3]/2)
            px_d = np.hypot(c10[0] - c22[0], c10[1] - c22[1])
            avg_h = 0.5 * (t10.current_bbox_xywh[3] + t22.current_bbox_xywh[3])
            norm_px = px_d / max(15.0, avg_h)
            if f_idx % 2 == 0 or iou > 0.0 or norm_px < 1.0:
                print(f"F{f_idx} T10 & T22: iou={iou:.3f}, norm_px={norm_px:.2f}, spds=({k10.speed_kmh:.1f}, {k22.speed_kmh:.1f}), accs=({k10.acceleration_mag:.1f}, {k22.acceleration_mag:.1f}), provs=({t10.provenance_history[-1] if t10.provenance_history else 'N'}, {t22.provenance_history[-1] if t22.provenance_history else 'N'})")
