import os, sys
sys.path.insert(0, os.path.abspath('.'))
import tools.evaluate_wrong_way_target as e
from modules.episode_consolidator import consolidate_candidates_into_episodes

cfg = e.load_config('road_surveillance/config/default_config.yaml')
yolo_model = e.YOLO(cfg.detection.model_path)
preprocessor = e.LowLightPreprocessor(mode='N3')
vocab_gate = e.VocabularyGate()
quality_gate = e.DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = e.MotionROIGate()
tracker = e.RobustSurveillanceTracker()
camera_cfg = e.CameraConfiguration.from_config_dict('KTGByI3N4G0_00', cfg.calibration.get('default'), False)
homography = e.PlanarHomography.from_camera_config(camera_cfg, (1280, 720))
lm = e.LaneManager.from_config(camera_cfg.lanes, homography)
geo = e.GeometryEngine(homography=homography, lane_manager=lm)
event_eng = e.EventEngine(config=cfg.events, is_camera_calibrated=False)
event_eng.set_resolution(1280, 720)

vpath = r'C:\Users\KALAKA~1\Downloads\archive\real_videos\KTGByI3N4G0_00.mp4'
ingestor = e.DatasetVideoIngestor(vpath)

all_candidates = []
for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    proc, _ = preprocessor.process(frame.image)
    yres = yolo_model.predict(source=proc, imgsz=640, conf=0.02, device='cpu', verbose=False)[0]
    dets = [{'bbox': [float(x) for x in b], 'conf': float(c), 'class_id': int(cl), 'class_name': yolo_model.names[int(cl)].lower()} for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())] if yres.boxes is not None else []
    admitted, _ = quality_gate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(admitted, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    cands = event_eng.process_frame(f, frame.pts_ns, 'KTGByI3N4G0_00', active, kin, conv, lm)
    for c in cands:
        all_candidates.append({
            'frame': f,
            'timestamp_s': f / 30.0,
            'event_type': c.event_type.value if hasattr(c.event_type, 'value') else str(c.event_type),
            'track_ids': c.track_ids,
            'confidence': float(c.confidence),
            'signals': c.signals,
            'description': c.description
        })

print(f"Total raw candidates: {len(all_candidates)}")
for i, c in enumerate(all_candidates):
    print(f"  [{i+1}] F{c['frame']} | {c['event_type']} | tracks={c['track_ids']} | conf={c['confidence']:.2f}")

episodes = consolidate_candidates_into_episodes(all_candidates, 'KTGByI3N4G0_00', cooldown_seconds=10.0, max_gap_frames=60, gt_frame=91, temporal_tolerance=45)
print(f"\nConsolidated episodes: {len(episodes)}")
for i, ep in enumerate(episodes):
    nm_cnt = len(ep.peak_candidate.get('signals', {}).get('near_misses', []))
    print(f"  Episode {i+1}: {ep.primary_event_type} | F{ep.start_frame}->F{ep.end_frame} (Peak F{ep.peak_frame}) | Tracks: {ep.involved_track_ids} | InGT: {ep.is_in_gt_window} | Near-misses: {nm_cnt}")
