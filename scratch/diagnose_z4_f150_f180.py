import os, sys, json
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from road_surveillance.config import load_config
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.episode_consolidator import consolidate_candidates_into_episodes, ConsolidatedEpisode, ActorDispatchTracker
from modules.vlm_arbitrator import GroqArbitrator, create_evidence_strip
from ultralytics import YOLO

video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(crop_resolution=320, crop_conf_threshold=0.05, max_crops_per_frame=1, min_crop_interval=2)
arbitrator = GroqArbitrator(mock_fallback=True)

cap_meta = cv2.VideoCapture(video_path)
vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
cap_meta.release()

raw_calib = cfg.calibration.get("Z4kg2Ev3vhk_00") or cfg.calibration.get("default")
camera_cfg = CameraConfiguration.from_config_dict("Z4kg2Ev3vhk_00", raw_calib, is_calibrated=False)
homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
geometry_engine = GeometryEngine(homography, lane_manager, False)
event_engine = EventEngine(cfg.events, False)
event_engine.set_resolution(vid_w, vid_h)
tracker = RobustSurveillanceTracker()

all_candidates = []
frame_detections = {}
frame_tracks_info = {}
frame_pair_evals = {}
frame_suppressed = {}

ingestor = DatasetVideoIngestor(video_path)
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 185:
        break
    img = frame.image
    proc_img, _ = preprocessor.process(img)
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    yres = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        b_xyxy = yres.boxes.xyxy.cpu().numpy()
        b_conf = yres.boxes.conf.cpu().numpy()
        b_cls = yres.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": yolo_model.names[int(bcl)].lower()
            })

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    recovered_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=raw_dets,
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
    )
    raw_dets.extend(recovered_micro)

    admitted_recs, _ = quality_gate.process_and_admit(raw_dets, f_idx, timestamp_s, (vid_h, vid_w), homography, roadway_polygon=road_poly)
    tracklets, track_dict = tracker.update(admitted_recs, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geometry_engine.process_frame(tracklets, track_dict)

    if 150 <= f_idx <= 180:
        # Record raw detections
        frame_detections[f_idx] = [
            {
                "bbox": [round(float(x), 1) for x in (getattr(d, "bbox_xyxy", getattr(d, "bbox", [])))],
                "conf": round(float(getattr(d, "confidence", getattr(d, "conf", 0.0))), 3),
                "cls": getattr(d, "class_label", getattr(d, "class_name", ""))
            }
            for d in admitted_recs
        ]
        # Record tracks
        frame_tracks_info[f_idx] = [
            {
                "id": t_id,
                "label": tr.class_label,
                "bbox": [round(x, 1) for x in tr.current_bbox_xywh],
                "age": tr.track_age,
                "status": tr.status,
                "speed_kmh": round(kinematics[t_id].speed_kmh, 1) if t_id in kinematics else 0.0,
                "heading_deg": round(kinematics[t_id].heading_deg, 1) if t_id in kinematics else 0.0,
                "accel_ms2": round(kinematics[t_id].acceleration_mag, 1) if t_id in kinematics else 0.0,
            }
            for t_id, tr in track_dict.items()
        ]

    # Pre-capture suppressed count
    col_det = [d for d in event_engine.detectors if hasattr(d, "suppressed_events")][0]
    supp_before = len(col_det.suppressed_events)

    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="Z4kg2Ev3vhk_00",
        tracks=track_dict,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )

    new_supp = col_det.suppressed_events[supp_before:]
    if 150 <= f_idx <= 180 and new_supp:
        frame_suppressed[f_idx] = new_supp

    for c in cands:
        etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
        all_candidates.append({
            "frame": f_idx,
            "timestamp_s": round(timestamp_s, 3),
            "event_type": etype,
            "track_ids": c.track_ids,
            "confidence": round(float(c.confidence), 2),
            "description": c.description,
            "signals": c.signals or {},
            "ground_coordinates": getattr(c, "ground_coordinates", (0.0, 0.0)),
        })

    # Detailed pair evaluation in F150-F180
    if 150 <= f_idx <= 180:
        pair_evals = []
        for conv in convergences:
            t1_id, t2_id = conv.track_id_1, conv.track_id_2
            if t1_id in track_dict and t2_id in track_dict:
                tr1 = track_dict[t1_id]
                tr2 = track_dict[t2_id]
                k1 = kinematics.get(t1_id)
                k2 = kinematics.get(t2_id)
                if k1 and k2 and tr1.class_label.lower() in col_det.VEHICLE_LABELS and tr2.class_label.lower() in col_det.VEHICLE_LABELS:
                    pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
                    dist = conv.distance_m
                    b1 = tr1.current_bbox_xywh
                    b2 = tr2.current_bbox_xywh
                    iou = col_det._compute_bbox_iou(b1, b2)
                    iou_x = col_det._compute_bbox_iou_x(b1, b2)
                    c1_x = b1[0] + b1[2] / 2.0
                    c1_y = b1[1] + b1[3] / 2.0
                    c2_x = b2[0] + b2[2] / 2.0
                    c2_y = b2[1] + b2[3] / 2.0
                    px_dist = float(np.hypot(c1_x - c2_x, c1_y - c2_y))
                    avg_h = 0.5 * (b1[3] + b2[3])
                    char_dim = max(15.0, avg_h)
                    norm_pixel_dist = px_dist / char_dim

                    cos_h = float(np.dot(k1.heading_vector, k2.heading_vector))
                    cos_clamped = max(-1.0, min(1.0, cos_h))
                    h_deg = float(np.degrees(np.arccos(cos_clamped)))

                    a1_long = k1.acceleration_mag
                    a2_long = k2.acceleration_mag
                    c_cnt = col_det.closure_persist_counts.get(pair_key, 0)
                    is_parallel = (h_deg <= 15.0 or abs(h_deg - 180.0) <= 15.0)
                    has_contact = (iou >= 0.15 or (iou >= 0.10 and iou_x >= 0.50) or (not col_det.is_camera_calibrated and norm_pixel_dist <= 0.50))
                    has_decel = (a1_long <= -6.0 or a2_long <= -6.0)
                    has_sanity = ((dist <= 3.5) if col_det.is_camera_calibrated else (norm_pixel_dist <= 1.0 or dist <= 4.0))

                    pair_evals.append({
                        "pair": [t1_id, t2_id],
                        "dist_m": round(dist, 2),
                        "dh_deg": round(h_deg, 1),
                        "iou": round(iou, 3),
                        "norm_pixel_dist": round(norm_pixel_dist, 3),
                        "accels": [round(a1_long, 1), round(a2_long, 1)],
                        "closure_cnt": c_cnt,
                        "is_parallel": is_parallel,
                        "has_contact": has_contact,
                        "has_decel": has_decel,
                        "has_sanity": has_sanity,
                    })
        frame_pair_evals[f_idx] = pair_evals

# 5. Episode Consolidation
episodes = consolidate_candidates_into_episodes(
    raw_candidates=all_candidates,
    video_id="Z4kg2Ev3vhk_00",
    cooldown_seconds=10.0,
    max_gap_frames=60,
    gt_frame=170,
    temporal_tolerance=45,
)

# 6. VLM arbitration on episodes
vlm_reports = []
for ep in episodes:
    cand = ep.peak_candidate
    cand_f = ep.peak_frame
    c_type = cand["event_type"]
    c_tracks = cand["track_ids"]
    c_sig = cand.get("signals", {})

    telemetry = {
        "video_id": "Z4kg2Ev3vhk_00",
        "candidate_frame": cand_f,
        "gt_impact_frame": 170,
        "event_type": c_type,
        "involved_tracks": c_tracks,
        "accel_ms2": float(c_sig.get("accel_1_ms2", c_sig.get("metric_accel_ms2", 0.0))),
        "dist_m": float(c_sig.get("distance_m", c_sig.get("ground_distance_m", 1.0))),
        "bbox_iou": float(c_sig.get("bbox_iou", 0.0)),
        "is_kinematic_closure_corridor": c_sig.get("is_kinematic_closure_corridor", False),
    }
    # Mock strip
    strip = np.zeros((336, 1008, 3), dtype=np.uint8)
    vlm_res = arbitrator.arbitrate(
        evidence_strip=strip,
        telemetry=telemetry,
        candidate_frame=cand_f,
        video_id="Z4kg2Ev3vhk_00",
    )
    ep.vlm_result = vlm_res
    ep.is_confirmed = (vlm_res.get("verdict") == "CONFIRMED_INCIDENT")
    vlm_reports.append({
        "ep_id": ep.episode_id,
        "peak_f": ep.peak_frame,
        "range": [ep.start_frame, ep.end_frame],
        "tracks": ep.involved_track_ids,
        "primary_pair": ep.primary_actor_pair,
        "in_window": ep.is_in_gt_window,
        "confirmed": ep.is_confirmed,
        "verdict": vlm_res.get("verdict"),
        "rationale": vlm_res.get("rationale"),
        "source": vlm_res.get("source"),
    })

output_data = {
    "detections_f150_f180": frame_detections,
    "tracks_f150_f180": frame_tracks_info,
    "pair_evals_f150_f180": frame_pair_evals,
    "suppressed_f150_f180": frame_suppressed,
    "candidates": all_candidates,
    "episodes": vlm_reports,
}

with open("scratch/z4_diagnostic_output.json", "w") as f:
    json.dump(output_data, f, indent=2)

print("SUCCESS: Diagnostic data written to scratch/z4_diagnostic_output.json")
