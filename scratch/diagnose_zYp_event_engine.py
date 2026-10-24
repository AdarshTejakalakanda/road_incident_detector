import sys, os
sys.path.insert(0, os.path.abspath('.'))

import logging
logging.basicConfig(level=logging.INFO)

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

event_engine = EventEngine(
    config=cfg.events,
    is_camera_calibrated=cam_cfg.is_calibrated,
)
event_engine.set_resolution(1280, 720)

tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

sv_det = None
coll_det = None
for d in event_engine.detectors:
    if type(d).__name__ == "SingleVehicleAnomalyDetector":
        sv_det = d
    elif type(d).__name__ == "CollisionCandidateDetector":
        coll_det = d

print("Running EventEngine diagnosis for zYp-honyJC0_00...")
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 80:
        break
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

    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="zYp-honyJC0_00",
        tracks=active_tracks,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_mgr,
    )
    if cands:
        print(f"!!! TRIGGER at F{f_idx} ({timestamp_s:.2f}s): {[c.candidate_type for c in cands]}")

    if sv_det and 7 in active_tracks and 40 <= f_idx <= 65:
        tr7 = active_tracks[7]
        k7 = kinematics.get(7)
        b = tr7.current_bbox_xywh
        char_dim = max(15.0, max(b[2], b[3]))
        px_spd = getattr(k7, "pixel_speed", 0.0)
        px_acc = getattr(k7, "pixel_acceleration", 0.0)
        m_spd = (px_spd / char_dim) * 4.5 * 3.6
        m_a = (px_acc / char_dim) * 4.5
        print(f"F{f_idx:03d} tr7: pos=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}), k_spd={k7.speed_kmh:.1f}, k_acc={k7.acceleration_mag:.1f}, m_spd={m_spd:.1f}, m_a={m_a:.1f}")

print("\n--- SV Suppressed Events ---")
for s in sv_det.suppressed_events:
    print(s)

print("\n--- Coll Suppressed Events ---")
for s in coll_det.suppressed_events:
    print(s)
