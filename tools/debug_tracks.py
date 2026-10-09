import os, sys
sys.path.insert(0, os.path.abspath('.'))
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO

vpath = r'C:\Users\KALAKA~1\Downloads\archive\real_videos\KTGByI3N4G0_00.mp4'
cfg = load_config('road_surveillance/config/default_config.yaml')
yolo = YOLO(cfg.detection.model_path)
prep = LowLightPreprocessor(mode='N3')
vgate = VocabularyGate()
qgate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vgate)
tracker = RobustSurveillanceTracker()
camera_cfg = CameraConfiguration.from_config_dict('KTGByI3N4G0_00', cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(camera_cfg, (1280, 720))
lm = LaneManager.from_config(camera_cfg.lanes, homography)
geo = GeometryEngine(homography=homography, lane_manager=lm)

ingestor = DatasetVideoIngestor(vpath)
for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    proc, _ = prep.process(frame.image)
    yres = yolo.predict(source=proc, imgsz=640, conf=0.02, device='cpu', verbose=False)[0]
    dets = [{'bbox': [float(x) for x in b], 'conf': float(c), 'class_id': int(cl), 'class_name': yolo.names[int(cl)].lower()} for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())] if yres.boxes is not None else []
    adm, _ = qgate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(adm, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    if f >= 35 and 11 in active:
        tr = active[11]
        k = kin.get(11)
        gc = tr.calculate_ground_contact(tr.current_bbox_xywh)
        h = k.heading_vector if k else None
        spd = k.speed_kmh if k else 0
        print(f'F{f:2d} Tid 11: conf={tr.detection_confidence:.2f} spd={spd:.1f} age={tr.track_age} gc=({gc[0]:.1f}, {gc[1]:.1f}) heading=({round(h[0],2) if h else 0}, {round(h[1],2) if h else 0})')
    if f > 55:
        break

