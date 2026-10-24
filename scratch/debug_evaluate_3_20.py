import sys, os
sys.path.insert(0, os.path.abspath('.'))
from road_surveillance.events.candidates.collision import CollisionCandidateDetector
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.geometry.engine import GeometryEngine
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO
from run_pipeline import resolve_video_path, load_config

video_path = resolve_video_path('zYgBOSbkY3A_00')
cfg = load_config('road_surveillance/config/default_config.yaml')
model = YOLO(cfg.detection.model_path)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)

cam_cfg = CameraConfiguration.from_config_dict('zYgBOSbkY3A_00', cfg.calibration.get('zYgBOSbkY3A_00') or cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (1920, 1080))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geo = GeometryEngine(homography, lane_mgr, False)

col_det = CollisionCandidateDetector(
    distance_decay_threshold_m=cfg.events.collision.distance_decay_threshold_m,
    tca_threshold_sec=cfg.events.collision.tca_threshold_sec,
    deceleration_threshold_ms2=cfg.events.collision.deceleration_threshold_ms2,
    post_proximity_distance_m=cfg.events.collision.post_proximity_distance_m,
    iou_overlap_threshold=cfg.events.collision.iou_overlap_threshold,
    is_camera_calibrated=False,
    min_persistence_frames=2,
    min_track_age=8,
)

road_poly = vocab_gate.get_default_roadway_polygon((1080, 1920))
tracker = RobustSurveillanceTracker()
ingestor = DatasetVideoIngestor(video_path)

print("Starting debug_evaluate_3_20...")
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

    if f_idx >= 55 and (3 in active_tracks and 20 in active_tracks):
        # Check if (3, 20) is in convergences
        conv = next((c for c in convergences if (c.track_id_1==3 and c.track_id_2==20) or (c.track_id_1==20 and c.track_id_2==3)), None)
        tr1 = active_tracks[3]
        tr2 = active_tracks[20]
        k1 = kinematics.get(3)
        k2 = kinematics.get(20)
        has_obs1 = tr1.has_real_observation_near(-1, window=3)
        has_obs2 = tr2.has_real_observation_near(-1, window=3)
        lbl1 = tr1.class_label
        lbl2 = tr2.class_label
        print(f"F{f_idx} (3,20): conv={conv is not None}, k1={k1 is not None}, k2={k2 is not None}, lbl1={lbl1}, lbl2={lbl2}, age1={tr1.track_age}, age2={tr2.track_age}, obs1={has_obs1}, obs2={has_obs2}")

    cands = col_det.evaluate(f_idx, pts_ns, 'zYgBOSbkY3A_00', active_tracks, kinematics, convergences, lane_mgr)
    for c in cands:
        print(f"CANDIDATE TRIGGERED F{f_idx}: {c.track_ids}")
