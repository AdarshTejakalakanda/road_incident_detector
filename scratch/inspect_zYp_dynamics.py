import sys, os
sys.path.insert(0, os.path.abspath('.'))
from run_pipeline import resolve_video_path, load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
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
tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Tracking details for zYp-honyJC0_00...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    res = model.predict(source=img, imgsz=(736, 1280), conf=0.02, device="cpu", verbose=False)[0]
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

    # Let's inspect tracks that have min_acc <= -20 or speed drop
    for tid, tr in active_tracks.items():
        k = kinematics.get(tid)
        if k and (k.acceleration_mag <= -20.0 or (k.speed_kmh < 5.0 and getattr(tr, 'max_recent_speed_kmh', 0.0) >= 25.0)):
            b = tr.current_bbox_xywh
            print(f"F{f_idx:03d} ({timestamp_s:.2f}s) Track {tid:2d} ({tr.get_effective_label()}): pos=({b[0]+b[2]/2:.0f},{b[1]+b[3]/2:.0f}), w={b[2]:.0f}, h={b[3]:.0f}, spd={k.speed_kmh:.1f}km/h, acc={k.acceleration_mag:.1f}m/s2, age={tr.track_age}")
