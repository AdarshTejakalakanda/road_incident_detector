import cv2, os, sys
sys.path.insert(0, os.path.abspath('.'))
from ultralytics import YOLO
from run_pipeline import resolve_video_path, load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate

video_path = resolve_video_path('zYgBOSbkY3A_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)

cam_cfg = CameraConfiguration.from_config_dict('zYgBOSbkY3A_00', cfg.calibration.get('zYgBOSbkY3A_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (1920, 1080))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)
event_engine = EventEngine(cfg.events)
event_engine.set_resolution(1920, 1080)
road_poly = vocab_gate.get_default_roadway_polygon((1080, 1920))

tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Processing zYgBOSbkY3A_00...")
detected_tracks_at_crash = []
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    # YOLO native
    res = model.predict(source=img, imgsz=(1088, 1920), conf=0.02, device="cpu", verbose=False)[0]
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
    cands = event_engine.process_frame(f_idx, pts_ns, 'zYgBOSbkY3A_00', active_tracks, kinematics, convergences, lane_mgr)

    # Let's inspect all active tracks in the upper center region (x: 500-1100, y: 50-350)
    for tid, tr in active_tracks.items():
        bx, by, bw, bh = tr.current_bbox_xywh
        cx, cy = bx + bw/2, by + bh/2
        if 500 <= cx <= 1100 and 50 <= cy <= 350:
            k = kinematics.get(tid)
            spd = getattr(k, 'speed_kmh', 0.0)
            acc = getattr(k, 'acceleration_mag', 0.0)
            print(f"F{f_idx} Crash-Zone Track {tid} ({tr.class_label}): center=({cx:.1f}, {cy:.1f}), bbox=({bx:.1f}, {by:.1f}, {bw:.1f}, {bh:.1f}), spd={spd:.1f}, acc={acc:.1f}")

    for c in cands:
        print(f"--> EVENT F{f_idx}: {c.event_type}, tracks={c.track_ids}, desc={c.description}")
