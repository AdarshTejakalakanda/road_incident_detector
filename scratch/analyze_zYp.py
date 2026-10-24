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

video_path = resolve_video_path('zYp-honyJC0_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)

cam_cfg = CameraConfiguration.from_config_dict('zYp-honyJC0_00', cfg.calibration.get('zYp-honyJC0_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (1280, 720))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)
event_engine = EventEngine(cfg.events, is_camera_calibrated=False)
event_engine.set_resolution(1280, 720)

road_poly = vocab_gate.get_default_roadway_polygon((720, 1280))
tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print(f"Analyzing all 222 frames of zYp-honyJC0_00...")
all_cands = []
all_tracks_summary = {}

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

    candidates = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id='zYp-honyJC0_00',
        tracks=active_tracks,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_mgr,
    )
    if candidates:
        for c in candidates:
            print(f"--> EVENT CANDIDATE F{f_idx} ({timestamp_s:.2f}s): {c.event_type}, tracks={c.track_ids}, conf={c.confidence}")
            all_cands.append(c)

    for tid, tr in active_tracks.items():
        k = kinematics.get(tid)
        if tid not in all_tracks_summary:
            all_tracks_summary[tid] = {
                "label": tr.get_effective_label(),
                "first_f": f_idx,
                "last_f": f_idx,
                "max_spd": 0.0,
                "min_acc": 0.0,
                "max_acc": 0.0,
                "positions": [],
            }
        summ = all_tracks_summary[tid]
        summ["last_f"] = f_idx
        if k:
            summ["max_spd"] = max(summ["max_spd"], k.speed_kmh)
            summ["min_acc"] = min(summ["min_acc"], k.acceleration_mag)
            summ["max_acc"] = max(summ["max_acc"], k.acceleration_mag)
            summ["positions"].append(k.ground_pos)

print("\n--- TRACKS SUMMARY ---")
for tid, s in sorted(all_tracks_summary.items()):
    dur = s["last_f"] - s["first_f"] + 1
    print(f"Track {tid:2d} ({s['label']}): frames={s['first_f']}-{s['last_f']} ({dur}f), max_spd={s['max_spd']:.1f}km/h, min_acc={s['min_acc']:.1f}m/s2, max_acc={s['max_acc']:.1f}m/s2")

print(f"\nTotal candidates: {len(all_cands)}")
