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
from modules.episode_consolidator import consolidate_candidates_into_episodes, ConsolidatedEpisode, ActorDispatchTracker
from modules.vlm_arbitrator import GroqArbitrator
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

print("Starting deep frame inspection...")
frame_records = {}

for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 185:
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
                "bbox": [round(float(x), 1) for x in bx],
                "conf": round(float(bc), 3),
                "cls": yolo_model.names[int(bcl)].lower()
            })

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    recovered_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=[{"bbox": d["bbox"], "conf": d["conf"], "class_id": 2, "class_name": d["cls"]} for d in raw_yolo_dets],
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
    )

    all_dets = list(raw_yolo_dets)
    for rm in recovered_micro:
        all_dets.append({
            "bbox": [round(float(x), 1) for x in rm.get("bbox", [])],
            "conf": round(float(rm.get("conf", 0.0)), 3),
            "cls": rm.get("class_name", "recovered"),
        })

    admitted_recs, q_tel = quality_gate.process_and_admit(
        [{"bbox": d["bbox"], "conf": d["conf"], "class_id": 2, "class_name": d["cls"]} for d in all_dets],
        f_idx, timestamp_s, (vid_h, vid_w), homography, roadway_polygon=road_poly
    )

    tracklets, track_dict = tracker.update(admitted_recs, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geometry_engine.process_frame(tracklets, track_dict)

    col_det = [d for d in event_engine.detectors if hasattr(d, "suppressed_events")][0]
    supp_before = len(col_det.suppressed_events)

    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="Z4kg2Ev3vhk_00",
        tracks=track_dict,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )
    new_supp = col_det.suppressed_events[supp_before:]

    if 150 <= f_idx <= 180:
        frame_records[f_idx] = {
            "raw_yolo": raw_yolo_dets,
            "motion_recovered": [
                {"bbox": [round(float(x), 1) for x in rm.get("bbox", [])], "conf": round(float(rm.get("conf", 0.0)), 3), "cls": rm.get("class_name", "")}
                for rm in recovered_micro
            ],
            "admitted_quality": [
                {"bbox": [round(float(x), 1) for x in r.bbox_xyxy], "conf": round(float(r.raw_confidence), 3), "trust": round(float(r.trust_score), 3), "cls": r.fused_class}
                for r in admitted_recs
            ],
            "tracks": [
                {
                    "id": tid,
                    "cls": tr.class_label,
                    "age": tr.track_age,
                    "status": tr.status,
                    "bbox": [round(float(x), 1) for x in tr.current_bbox_xywh],
                    "provenance": tr.provenance_history[-1] if tr.provenance_history else "NONE",
                    "spd": round(float(kinematics[tid].speed_kmh), 1) if tid in kinematics else 0.0,
                    "hdg": round(float(kinematics[tid].heading_deg), 1) if tid in kinematics else 0.0,
                    "accel": round(float(kinematics[tid].acceleration_mag), 1) if tid in kinematics else 0.0,
                }
                for tid, tr in track_dict.items()
            ],
            "candidates": [
                {
                    "type": c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type),
                    "tracks": c.track_ids,
                    "conf": round(float(c.confidence), 2),
                    "signals": c.signals or {},
                }
                for c in cands
            ],
            "suppressed": new_supp,
        }

with open("scratch/z4_deep_records.json", "w") as f:
    json.dump(frame_records, f, indent=2)

print("Saved scratch/z4_deep_records.json")
