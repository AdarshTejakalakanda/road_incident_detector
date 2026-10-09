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
from road_surveillance.events.engine import EventEngine
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
event_eng = EventEngine(config=cfg.events, is_camera_calibrated=False)
event_eng.set_resolution(1280, 720)
ww_mod = event_eng.wrong_way_detector.module

ingestor = DatasetVideoIngestor(vpath)
for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    if f > 35:
        break
    proc, _ = prep.process(frame.image)
    yres = yolo.predict(source=proc, imgsz=640, conf=0.02, device='cpu', verbose=False)[0]
    dets = [{'bbox': [float(x) for x in b], 'conf': float(c), 'class_id': int(cl), 'class_name': yolo.names[int(cl)].lower()} for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())] if yres.boxes is not None else []
    adm, _ = qgate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(adm, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    cands = event_eng.process_frame(f, frame.pts_ns, 'KTGByI3N4G0_00', active, kin, conv, lm)
    if f in [10, 20, 25, 27, 30]:
        print(f'=== FRAME {f} ===')
        print('Persisted cell models:', list(ww_mod.flow_field._persisted_models.keys()))
        for cell, m in ww_mod.flow_field._persisted_models.items():
            print(f'  Cell {cell}: dir=({m.direction_vector[0]:.2f}, {m.direction_vector[1]:.2f}), coh={m.coherence:.2f}, count={m.sample_count}, tids={ww_mod.flow_field._cell_track_ids.get(cell)}')
        if 6 in active:
            tr6 = active[6]
            k6 = kin.get(6)
            gc6 = tr6.calculate_ground_contact(tr6.current_bbox_xywh)
            cell6 = ww_mod.flow_field._get_cell_coord(gc6[0]/1280.0, gc6[1]/720.0)
            fm6 = ww_mod.flow_field.get_flow_model_for_position(gc6[0]/1280.0, gc6[1]/720.0)
            print(f'  Track 6: cell={cell6}, spd={k6.speed_kmh if k6 else 0:.1f}, heading={k6.heading_vector if k6 else None}, vc={ww_mod.violation_counts.get(6, 0)}, fm={fm6 is not None}')
