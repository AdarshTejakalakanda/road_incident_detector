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
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO

video_path = resolve_video_path('zYgBOSbkY3A_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)

cam_cfg = CameraConfiguration.from_config_dict('zYgBOSbkY3A_00', cfg.calibration.get('zYgBOSbkY3A_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (1920, 1080))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)
event_engine = EventEngine(cfg.events, is_camera_calibrated=False)
event_engine.set_resolution(1920, 1080)

road_poly = vocab_gate.get_default_roadway_polygon((1080, 1920))
tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Starting diagnostics on zYgBOSbkY3A_00...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    img = frame.image
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

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

    candidates = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id='zYgBOSbkY3A_00',
        tracks=active_tracks,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_mgr,
    )

    if candidates:
        for c in candidates:
            print(f"[CANDIDATE F{f_idx:03d}] type={c.event_type}, tracks={c.track_ids}, conf={c.confidence:.2f}")

    # Inspect tracks near intersection (X: 650-1150, Y: 80-350)
    for tid, tr in active_tracks.items():
        bx = tr.current_bbox_xywh
        cx = bx[0] + bx[2] / 2
        cy = bx[1] + bx[3] / 2
        if 650 <= cx <= 1150 and 80 <= cy <= 350:
            k = kinematics.get(tid)
            spd = k.speed_kmh if k else 0.0
            acc = k.acceleration_mag if k else 0.0
            lbl = tr.get_effective_label()
            if f_idx >= 60:
                print(f"  F{f_idx:03d} Track {tid:2d} ({lbl}): pos=({cx:.0f},{cy:.0f}), w={bx[2]:.0f}, h={bx[3]:.0f}, spd={spd:.1f}km/h, acc={acc:.1f}m/s2, age={tr.track_age}")

    # Inspect convergences involving tracks in this intersection area
    if f_idx >= 60:
        for conv in convergences:
            t1, t2 = conv.track_id_1, conv.track_id_2
            tr1, tr2 = active_tracks.get(t1), active_tracks.get(t2)
            if tr1 and tr2:
                b1, b2 = tr1.current_bbox_xywh, tr2.current_bbox_xywh
                c1x = b1[0] + b1[2]/2
                c2x = b2[0] + b2[2]/2
                if 650 <= c1x <= 1150 and 650 <= c2x <= 1150:
                    print(f"  --> Convergence F{f_idx} tracks ({t1},{t2}): dist_m={conv.distance_m:.1f}, tca={conv.tca_sec:.1f}s")
