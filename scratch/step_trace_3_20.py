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
import numpy as np

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

orig_log = col_det.log_suppressed
def my_log(frame_idx, tracks, reason, dist_or_dca):
    if (3 in tracks and 20 in tracks):
        print(f"--> LOG_SUPPRESSED F{frame_idx}: reason={reason}, val={dist_or_dca}", flush=True)
    return orig_log(frame_idx, tracks, reason, dist_or_dca)
col_det.log_suppressed = my_log

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

    if f_idx in (65, 66, 67, 68, 69, 70):
        print(f"=== FRAME {f_idx} ===", flush=True)
        conv = next((c for c in convergences if (c.track_id_1==3 and c.track_id_2==20) or (c.track_id_1==20 and c.track_id_2==3)), None)
        if conv and (3 in active_tracks and 20 in active_tracks):
            tr1 = active_tracks[3]
            tr2 = active_tracks[20]
            k1 = kinematics.get(3)
            k2 = kinematics.get(20)
            iou = col_det._compute_bbox_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            c1_x = b1[0] + b1[2] / 2.0
            c1_y = b1[1] + b1[3] / 2.0
            c2_x = b2[0] + b2[2] / 2.0
            c2_y = b2[1] + b2[3] / 2.0
            px_dist = float(np.hypot(c1_x - c2_x, c1_y - c2_y))
            avg_h = 0.5 * (b1[3] + b2[3])
            char_dim = max(15.0, avg_h)
            norm_pixel_dist = px_dist / char_dim
            print(f"  conv: dist_m={conv.distance_m:.2f}, iou={iou:.3f}, norm_px_dist={norm_pixel_dist:.3f}, k1_spd={k1.speed_kmh:.1f}, k2_spd={k2.speed_kmh:.1f}, k1_acc={k1.acceleration_mag:.1f}, k2_acc={k2.acceleration_mag:.1f}", flush=True)

    cands = col_det.evaluate(f_idx, pts_ns, 'zYgBOSbkY3A_00', active_tracks, kinematics, convergences, lane_mgr)
    for c in cands:
        if 3 in c.track_ids and 20 in c.track_ids:
            print(f"*** CANDIDATE DETECTED (3, 20) @ F{f_idx} ***", flush=True)
