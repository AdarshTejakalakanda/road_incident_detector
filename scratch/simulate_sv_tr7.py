import sys, os
sys.path.insert(0, os.path.abspath('.'))
from run_pipeline import resolve_video_path, load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector, compute_single_vehicle_evidence, check_has_lead_vehicle
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO

video_path = resolve_video_path('zYp-honyJC0_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
cam_cfg = CameraConfiguration.from_config_dict('zYp-honyJC0_00', cfg.calibration.get('zYp-honyJC0_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (1280, 720))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)
road_poly = vocab_gate.get_default_roadway_polygon((720, 1280))

sv = SingleVehicleAnomalyDetector(
    is_camera_calibrated=False,
    video_warmup_frames=10,
    min_track_age=8,
)
sv.set_resolution(1280, 720)
tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Starting simulation...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 65:
        break
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    res = model.predict(source=img, imgsz=(736, 1280), conf=0.02, device='cpu', verbose=False)[0]
    raw_dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        b_xyxy = res.boxes.xyxy.cpu().numpy()
        b_conf = res.boxes.conf.cpu().numpy()
        b_cls = res.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            raw_dets.append({'bbox': [float(x) for x in bx], 'conf': float(bc), 'class_id': int(bcl), 'class_name': model.names[int(bcl)].lower()})

    admitted, _ = quality_gate.process_and_admit(raw_dets, f_idx, timestamp_s, img.shape[:2], homography, road_poly)
    tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geo.process_frame(tracklets, active_tracks)

    if 7 in active_tracks and 40 <= f_idx <= 60:
        tr7 = active_tracks[7]
        k7 = kinematics.get(7)
        has_lead = check_has_lead_vehicle(7, active_tracks, kinematics)
        s_single, detail = compute_single_vehicle_evidence(
            track=tr7,
            roadway_poly=road_poly,
            homography=homography,
            kinematics=k7,
            has_lead_vehicle=has_lead,
        )
        hist = sv.track_history.get(7, [])
        plausible, rej, tel = sv.evaluate_kinematic_plausibility(tr7, hist, -detail['decel_mag'], k7.speed_kmh, f_idx, pts_ns)
        print(f"F{f_idx:02d} ({timestamp_s:.2f}s) Tr7: s_single={s_single:.3f}, decel_mag={detail['decel_mag']:.1f}, plau={plausible}, rej={rej}, v_entry={tel.get('v_entry_kmh')}")

    cands = sv.evaluate(f_idx, pts_ns, 'zYp-honyJC0_00', active_tracks, kinematics, convergences, lane_mgr)
    if cands:
        for c in cands:
            print(f"  >>> CANDIDATE: F{f_idx} type={c.event_type} trk={c.track_ids} signals={c.signals.get('anomaly_type')}")
