import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import cv2
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from road_surveillance.events.engine import EventEngine
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from ultralytics import YOLO

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\KTGByI3N4G0_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo = YOLO(cfg.detection.model_path)
prep = LowLightPreprocessor(mode="N3")
vgate = VocabularyGate()
qgate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vgate)
mgate = MotionROIGate()
tracker = RobustSurveillanceTracker()
camera_cfg = CameraConfiguration.from_config_dict("KTGByI3N4G0_00", cfg.calibration.get("default"), False)
homography = PlanarHomography.from_camera_config(camera_cfg, (1280, 720))
lm = LaneManager.from_config(camera_cfg.lanes, homography)
geo = GeometryEngine(homography=homography, lane_manager=lm)
event_eng = EventEngine(config=cfg.events, is_camera_calibrated=False)
event_eng.set_resolution(1280, 720)

ingestor = DatasetVideoIngestor(vpath)
for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    if f > 120:
        break
    proc, _ = prep.process(frame.image)
    yres = yolo.predict(source=proc, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    dets = [
        {"bbox": [float(x) for x in b], "conf": float(c), "class_id": int(cl), "class_name": yolo.names[int(cl)].lower()}
        for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())
    ] if yres.boxes is not None else []
    adm, _ = qgate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(adm, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    cands = event_eng.process_frame(f, frame.pts_ns, "KTGByI3N4G0_00", active, kin, conv, lm)
    
    ww_mod = event_eng.wrong_way_detector.module if event_eng.wrong_way_detector else None
    
    if 11 in active:
        tr11 = active[11]
        k11 = kin.get(11)
        vc = ww_mod.violation_counts.get(11, 0) if ww_mod else 0
        is_ww = getattr(tr11, "is_wrong_way", False)
        gc = tr11.calculate_ground_contact(tr11.current_bbox_xywh)
        fm = ww_mod.flow_field.get_flow_model_for_position(gc[0]/1280.0, gc[1]/720.0) if ww_mod else None
        fm_dir = fm.direction_vector if fm else None
        fm_coh = fm.coherence if fm else 0.0
        print(f"F{f:3d} | Tr11 age={tr11.track_age} spd={k11.speed_kmh if k11 else 0:.1f} heading={k11.heading_vector if k11 else None} vc={vc} is_ww={is_ww} flow_coh={fm_coh:.2f} flow_dir={fm_dir}")
    
    if cands:
        for c in cands:
            print(f"   >>> CANDIDATE @ F{f}: {c.event_type} | tracks={c.track_ids} | {c.description}")
