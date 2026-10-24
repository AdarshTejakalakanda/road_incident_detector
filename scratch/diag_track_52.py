import sys, os
sys.path.insert(0, os.path.abspath('.'))
from run_pipeline import resolve_video_path, load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO
import cv2

video_path = resolve_video_path('5uh8-3q51uI_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)

cam_cfg = CameraConfiguration.from_config_dict('5uh8-3q51uI_00', cfg.calibration.get('5uh8-3q51uI_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (960, 720))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)
event_engine = EventEngine(cfg.events)
event_engine.set_resolution(960, 720)
road_poly = vocab_gate.get_default_roadway_polygon((720, 960))
for d in event_engine.detectors:
    if isinstance(d, SingleVehicleAnomalyDetector):
        d.tau_single = 0.65
        d.set_roadway_polygon(road_poly)

tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Streaming...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 250:
        break
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)
    res = model.predict(source=img, imgsz=(736, 960), conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        b_xyxy = res.boxes.xyxy.cpu().numpy()
        b_conf = res.boxes.conf.cpu().numpy()
        b_cls = res.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": model.names[int(bcl)].lower(),
            })
    admitted, _ = quality_gate.process_and_admit(raw_dets, f_idx, timestamp_s, img.shape[:2], homography, road_poly)
    tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geo.process_frame(tracklets, active_tracks)
    cands = event_engine.process_frame(f_idx, pts_ns, '5uh8-3q51uI_00', active_tracks, kinematics, convergences, lane_mgr)
    if 52 in active_tracks and f_idx >= 238:
        tr52 = active_tracks[52]
        k52 = kinematics.get(52)
        print(f"F{f_idx} track 52: bbox={tr52.current_bbox_xywh}, spd={getattr(k52, 'speed_kmh', 0.0):.1f}, acc={getattr(k52, 'acceleration_mag', 0.0):.1f}, age={getattr(tr52, 'track_age', 0)}")
    for c in cands:
        if 52 in c.track_ids:
            print(f"--> ALERT on track 52 at F{f_idx}: {c.event_type}, signals={c.signals}")
