import os, sys
sys.path.insert(0, os.path.abspath('.'))
import tools.evaluate_wrong_way_target as e

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

for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    if f > 115:
        break
    proc, _ = preprocessor.process(frame.image)
    yres = yolo_model.predict(source=proc, imgsz=640, conf=0.02, device='cpu', verbose=False)[0]
    dets = [{'bbox': [float(x) for x in b], 'conf': float(c), 'class_id': int(cl), 'class_name': yolo_model.names[int(cl)].lower()} for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())] if yres.boxes is not None else []
    admitted, _ = quality_gate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(admitted, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    cands = event_eng.process_frame(f, frame.pts_ns, 'KTGByI3N4G0_00', active, kin, conv, lm)
    
    if f in [38, 39, 44, 45, 50, 70, 99, 107]:
        print(f"=== FRAME {f} ===")
        if 19 in active:
            tr19 = active[19]
            k19 = kin.get(19)
            gc19 = tr19.calculate_ground_contact(tr19.current_bbox_xywh)
            fm19 = event_eng.wrong_way_detector.module.flow_field.get_flow_model_for_position(gc19[0]/1280.0, gc19[1]/720.0)
            cell19 = event_eng.wrong_way_detector.module.flow_field._get_cell_coord(gc19[0]/1280.0, gc19[1]/720.0)
            print(f"  Tr19: cell={cell19} gc={gc19} spd={k19.speed_kmh if k19 else 0:.1f} heading={k19.heading_vector if k19 else None} fm={fm19 is not None} fm_dir={fm19.direction_vector if fm19 else None}")
        if 28 in active:
            tr28 = active[28]
            k28 = kin.get(28)
            gc28 = tr28.calculate_ground_contact(tr28.current_bbox_xywh)
            fm28 = event_eng.wrong_way_detector.module.flow_field.get_flow_model_for_position(gc28[0]/1280.0, gc28[1]/720.0)
            cell28 = event_eng.wrong_way_detector.module.flow_field._get_cell_coord(gc28[0]/1280.0, gc28[1]/720.0)
            print(f"  Tr28: label={tr28.get_effective_label()} fused={getattr(tr28, 'fused_class', '')} cell={cell28} gc={gc28} spd={k28.speed_kmh if k28 else 0:.1f}")
        for c in cands:
            print(f"  >>> CAND @ F{f}: {c.event_type} tracks={c.track_ids} signals={c.signals}")
