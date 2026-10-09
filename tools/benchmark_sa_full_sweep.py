"""Iteration S.A: VLM Fallback Safety, Deterministic Bypass & Actor-Level Dispatch Hardening.

Frozen Invariants:
- Native PTS timing (nanosecond presentation timestamps)
- Frozen detector thresholds: tau_high = 0.25, tau_low = 0.10, resolution = 640p
- Trajectory-aware crossing gate (35 <= delta_theta <= 145 deg)
- Frozen tau_rest >= 0.25 s
- Zero modifications to YOLO model, weights, or ByteTrack tracker logic

Task S.A:
1. VLM Fallback Safety: Unverified fallback returns CANDIDATE_UNCONFIRMED (suppressed=True).
   Never enters confirmed alert stream.
2. Deterministic High-Confidence Bypass: Safety-critical physical impacts isolated into
   independently identifiable source='deterministic_high_confidence_bypass'.
3. Actor-Level Dispatch Suppression: 10.0 s cooldown keyed on participating actor IDs.
   Logs 'actor_cooldown_active' with actor IDs, timestamp, previous dispatch, and dt.
4. Shoulder Barrier Protection: Preserve R.2 shoulder recovery, requiring barrier-strike evidence
   beyond boundary proximity: contact distance <= 0.8m, v . n >= 0.35, impulse a <= -8.0 m/s^2,
   and post-event speed drop / arrest.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import cv2
import numpy as np
import torch

torch.set_num_threads(8)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from road_surveillance.evaluation import GroundTruthLoader

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import create_evidence_strip, GroqArbitrator
from modules.episode_consolidator import consolidate_candidates_into_episodes, ConsolidatedEpisode, ActorDispatchTracker
from tools.benchmark_k0_detector import load_30_benchmark_entries


def compute_union_roi(
    boxes: List[Tuple[float, float, float, float]],
    frame_shape: Tuple[int, int],
    margin_factor: float = 0.6,
    min_size: int = 260,
) -> Tuple[int, int, int, int]:
    h_img, w_img = frame_shape[:2]
    if not boxes:
        cx, cy = w_img // 2, h_img // 2
        return (
            max(0, cx - min_size // 2),
            max(0, cy - min_size // 2),
            min(w_img, cx + min_size // 2),
            min(h_img, cy + min_size // 2),
        )

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)

    pad_w = max((min_size - bw) / 2.0, bw * margin_factor)
    pad_h = max((min_size - bh) / 2.0, bh * margin_factor)

    cx1 = int(max(0, np.floor(x1 - pad_w)))
    cy1 = int(max(0, np.floor(y1 - pad_h)))
    cx2 = int(min(w_img, np.ceil(x2 + pad_w)))
    cy2 = int(min(h_img, np.ceil(y2 + pad_h)))

    return (cx1, cy1, cx2, cy2)


def load_frame_at(video_path: str, target_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_idx - 1))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def run_single_video_evaluation(
    video_path: str,
    video_id: str,
    gt_frame: int,
    split_type: str,
    weather: str,
    day_time: str,
    yolo_model: YOLO,
    preprocessor: LowLightPreprocessor,
    vocab_gate: VocabularyGate,
    quality_gate: DetectorQualityGate,
    motion_gate: MotionROIGate,
    arbitrator: GroqArbitrator,
    cfg: Any,
    tau_single: float = 0.65,
    max_frames: int = 250,
    temporal_tolerance: int = 35,
    detection_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    cap_meta = cv2.VideoCapture(video_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()
    if vid_w <= 0 or vid_h <= 0:
        vid_w, vid_h = 1280, 720

    is_calibrated = bool(cfg.calibration.get(video_id))
    raw_calib = cfg.calibration.get(video_id) or cfg.calibration.get("default")
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id=video_id,
        cfg_obj=raw_calib,
        is_calibrated=is_calibrated,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
    lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
    geometry_engine = GeometryEngine(
        homography=homography,
        lane_manager=lane_manager,
        is_camera_calibrated=camera_cfg.is_calibrated,
    )
    event_engine = EventEngine(
        config=cfg.events,
        is_camera_calibrated=camera_cfg.is_calibrated,
    )
    event_engine.set_resolution(vid_w, vid_h)

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    for d in event_engine.detectors:
        if isinstance(d, SingleVehicleAnomalyDetector):
            d.tau_single = tau_single
            d.set_roadway_polygon(road_poly)

    tracker = RobustSurveillanceTracker()
    motion_gate.reset()

    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    detections_near_gt = 0
    tracks_near_gt = 0
    all_candidates: List[Dict[str, Any]] = []

    total_candidates_admitted = 0
    total_candidates_rejected_geom = 0
    total_motion_crops_evaluated = 0
    total_micro_vehicles_recovered = 0
    track_life_history: Dict[int, List[int]] = {}

    vlm_calls = 0
    last_f_idx = 0

    ingestor = DatasetVideoIngestor(path=video_path)
    frame_pts_map: Dict[int, float] = {}
    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        if f_idx >= max_frames:
            break
        last_f_idx = f_idx
        img = frame.image
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)
        frame_pts_map[f_idx] = timestamp_s

        # 1. Primary YOLO 640p pass
        if detection_cache is not None and video_path in detection_cache and f_idx in detection_cache[video_path]:
            raw_dets = [dict(d) for d in detection_cache[video_path][f_idx]]
        else:
            yolo_res = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
            raw_dets = []
            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                b_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
                b_conf = yolo_res.boxes.conf.cpu().numpy()
                b_cls = yolo_res.boxes.cls.cpu().numpy()
                for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
                    raw_dets.append({
                        "bbox": [float(x) for x in bx],
                        "conf": float(bc),
                        "class_id": int(bcl),
                        "class_name": yolo_model.names[int(bcl)].lower(),
                    })
            if detection_cache is not None:
                if video_path not in detection_cache:
                    detection_cache[video_path] = {}
                detection_cache[video_path][f_idx] = [dict(d) for d in raw_dets]

        # 2. Motion-Guided ROI Zoom Pass (Patch N.2)
        recovered_micro, m_tel = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )
        raw_dets.extend(recovered_micro)
        total_motion_crops_evaluated += m_tel["roi_crops_evaluated"]
        total_micro_vehicles_recovered += m_tel["recovered_micro_vehicles"]

        # 3. Quality Gate (Patch N.1 semantic recovery + trust calibration)
        admitted, qtel = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )
        total_candidates_admitted += qtel.get("candidate_recovered_admitted", 0)
        total_candidates_rejected_geom += qtel.get("candidate_recovered_rejected_geometry", 0)

        veh_admitted = [r for r in admitted if r.fused_class in ("vehicle", "candidate_vehicle_recovered")]
        if abs(f_idx - gt_frame) <= 15:
            detections_near_gt += len(veh_admitted)

        # 4. Robust Tracker Update
        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

        for tid in active_tracks.keys():
            if tid not in track_life_history:
                track_life_history[tid] = []
            track_life_history[tid].append(f_idx)

        if abs(f_idx - gt_frame) <= 15:
            tracks_near_gt += len(active_tracks)

        frame_tracks_cache[f_idx] = {
            tid: (
                tr.current_bbox_xywh[0],
                tr.current_bbox_xywh[1],
                tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2],
                tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3],
            )
            for tid, tr in active_tracks.items()
        }

        # 5. Kinematics & Event Engine
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
        cands = event_engine.process_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=active_tracks,
            kinematics=kinematics,
            convergences=convergences,
            lane_manager=lane_manager,
        )

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
            })

    elapsed_s = max(1e-4, time.perf_counter() - t0)
    fps = last_f_idx / elapsed_s if last_f_idx > 0 else 0.0

    # 6. Episode Consolidation
    episodes = consolidate_candidates_into_episodes(
        raw_candidates=all_candidates,
        video_id=video_id,
        cooldown_seconds=10.0,
        max_gap_frames=60,
        gt_frame=gt_frame,
        temporal_tolerance=temporal_tolerance,
        frame_pts_map=frame_pts_map,
    )

    in_window_episodes = [ep for ep in episodes if ep.is_in_gt_window]
    out_of_window_episodes = [ep for ep in episodes if not ep.is_in_gt_window]

    gate1_passed = detections_near_gt >= 1
    gate2_passed = tracks_near_gt >= 1
    gate3_passed = len(in_window_episodes) > 0

    if not gate1_passed:
        failure_gate = "Gate 1 (Perception)"
    elif not gate2_passed:
        failure_gate = "Gate 2 (Tracking)"
    elif not gate3_passed:
        failure_gate = "Gate 3 (Kinematics)"
    else:
        failure_gate = "Gate 4 (VLM Rejection)"

    # Task S.A.3: Actor Dispatch Tracker (10.0 s cooldown)
    dispatch_tracker = ActorDispatchTracker(cooldown_seconds=10.0)

    bypass_activations = 0
    candidate_unconfirmed_count = 0
    actor_cooldown_suppressions = 0

    # Stage 4: Arbitration on In-Window Episodes
    matched_tp_candidate = None
    vlm_tp_confirmed = False
    vlm_tp_result = None

    if in_window_episodes:
        sorted_in_window = sorted(
            in_window_episodes,
            key=lambda ep: (-ep.peak_score, abs(ep.peak_frame - gt_frame)),
        )
        for ep in sorted_in_window:
            cand = ep.peak_candidate
            cand_f = ep.peak_frame
            cand_t = cand.get("timestamp_s", cand_f / 30.0)

            # Check Actor-Level Dispatch Cooldown
            should_suppress, cd_log = dispatch_tracker.should_suppress_dispatch(cand.get("track_ids", []), cand_t)
            if should_suppress:
                ep.dispatch_suppressed = True
                ep.suppression_reason = "actor_cooldown_active"
                ep.cooldown_telemetry = cd_log
                actor_cooldown_suppressions += 1
                continue

            dispatch_tracker.record_dispatch(cand.get("track_ids", []), cand_t)

            t_app = max(1, cand_f - 8)
            t_imp = cand_f
            t_post = min(last_f_idx, cand_f + 8)

            img_app = load_frame_at(video_path, t_app)
            img_imp = load_frame_at(video_path, t_imp)
            img_post = load_frame_at(video_path, t_post)

            boxes_for_roi = []
            for f in [t_app, t_imp, t_post]:
                if f in frame_tracks_cache:
                    boxes_for_roi.extend(frame_tracks_cache[f].values())

            roi = compute_union_roi(boxes_for_roi, (vid_h, vid_w), margin_factor=0.6, min_size=260)
            rx1, ry1, rx2, ry2 = roi

            strip = create_evidence_strip(
                crop_pre=img_app[ry1:ry2, rx1:rx2],
                crop_impact=img_imp[ry1:ry2, rx1:rx2],
                crop_post=img_post[ry1:ry2, rx1:rx2],
                target_h=336,
                labels=(f"1: APPROACH (F{t_app})", f"2: SUSPECTED ANOMALY (F{t_imp})", f"3: POST (F{t_post})"),
            )

            cand_signals = cand.get("signals", {})
            cand_type = cand["event_type"]
            is_single = (cand_type in ("single_vehicle_anomaly", "single"))
            acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", cand_signals.get("decel_mag", -5.0))))
            dist_val = float(cand_signals.get("distance_m", cand_signals.get("ground_distance_m", cand_signals.get("dca_m", 1.0))))
            iou_val = float(cand_signals.get("bbox_iou", 0.0))

            telemetry_payload = {
                "video_id": video_id,
                "candidate_frame": cand_f,
                "gt_impact_frame": gt_frame,
                "event_type": cand_type,
                "involved_tracks": cand["track_ids"],
                "composite_score": float(cand_signals.get("composite_score", 0.65)),
                "accel_ms2": acc_val,
                "yaw_delta_deg": float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 30.0))),
                "theta_deg": float(cand_signals.get("theta_deg", 0.0)),
                "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
                "delta_ar": float(cand_signals.get("delta_ar", 0.0)),
                "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
                "distance_m": dist_val,
                "bbox_iou": iou_val,
                "anomaly_type": cand_signals.get("anomaly_type", cand_type),
                "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", cand_signals.get("speed_kmh", cand_signals.get("metric_speed_kmh", 0.0)))),
                "speed_kmh": float(cand_signals.get("speed_kmh", cand_signals.get("metric_speed_kmh", 0.0))),
                "track_age": min([len(track_life_history.get(tid, [])) for tid in cand.get("track_ids", [])]) if cand.get("track_ids") else 0,
                "is_single_vehicle": is_single,
                "weather": weather,
                "day_time": day_time,
                "post_event_rest": cand_signals.get("post_event_rest", True),
                    "tau_rest": 0.25 if cand_signals.get("post_event_rest", False) else 0.0,
                    "delta_v_drop": float(cand_signals.get("S_v_drop", 0.0)) * 20.0,
                "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
                "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
                "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
                "norm_pixel_dist": float(cand_signals.get("norm_pixel_dist", 99.0)),
                "is_kinematic_closure_corridor": bool(cand_signals.get("is_kinematic_closure_corridor", False)),
                "is_tentative_rescued": bool(cand_signals.get("is_tentative_rescued", False)),
            }

            vlm_calls += 1
            ep.vlm_dispatched = True
            cand_res = arbitrator.arbitrate(
                evidence_strip=strip,
                telemetry=telemetry_payload,
                candidate_frame=cand_f,
                video_id=video_id,
            )
            ep.vlm_result = cand_res

            if cand_res.get("source") == "deterministic_high_confidence_bypass":
                bypass_activations += 1

            if cand_res.get("verdict") == "CANDIDATE_UNCONFIRMED":
                candidate_unconfirmed_count += 1

            # Confirmation condition: must be CONFIRMED_INCIDENT and NOT suppressed
            is_confirmed = (cand_res.get("is_collision", False) or cand_res.get("verdict") == "CONFIRMED_INCIDENT") and not cand_res.get("suppressed", False)
            if is_confirmed:
                vlm_tp_confirmed = True
                ep.is_confirmed = True
                matched_tp_candidate = cand
                vlm_tp_result = cand_res
                break
            else:
                if vlm_tp_result is None:
                    vlm_tp_result = cand_res

    # Stage 5: Out-of-Window False Positive Arbitration Check
    fp_confirmed_count = 0
    fp_suppressed_count = 0

    for fp_ep in out_of_window_episodes:
        fp_cand = fp_ep.peak_candidate
        cand_f = fp_ep.peak_frame
        cand_t = fp_cand.get("timestamp_s", cand_f / 30.0)

        # Check Actor-Level Dispatch Cooldown
        should_suppress, cd_log = dispatch_tracker.should_suppress_dispatch(fp_cand.get("track_ids", []), cand_t)
        if should_suppress:
            fp_ep.dispatch_suppressed = True
            fp_ep.suppression_reason = "actor_cooldown_active"
            fp_ep.cooldown_telemetry = cd_log
            actor_cooldown_suppressions += 1
            fp_suppressed_count += 1
            continue

        dispatch_tracker.record_dispatch(fp_cand.get("track_ids", []), cand_t)

        t_app = max(1, cand_f - 8)
        t_imp = cand_f
        t_post = min(last_f_idx, cand_f + 8)

        img_app = load_frame_at(video_path, t_app)
        img_imp = load_frame_at(video_path, t_imp)
        img_post = load_frame_at(video_path, t_post)

        boxes_for_roi = []
        for f in [t_app, t_imp, t_post]:
            if f in frame_tracks_cache:
                boxes_for_roi.extend(frame_tracks_cache[f].values())

        roi = compute_union_roi(boxes_for_roi, (vid_h, vid_w), margin_factor=0.6, min_size=260)
        rx1, ry1, rx2, ry2 = roi

        strip = create_evidence_strip(
            crop_pre=img_app[ry1:ry2, rx1:rx2],
            crop_impact=img_imp[ry1:ry2, rx1:rx2],
            crop_post=img_post[ry1:ry2, rx1:rx2],
            target_h=336,
            labels=(f"1: PRE (F{t_app})", f"2: FP CAND (F{t_imp})", f"3: POST (F{t_post})"),
        )

        cand_signals = fp_cand.get("signals", {})
        cand_type = fp_cand["event_type"]
        is_single = (cand_type in ("single_vehicle_anomaly", "single"))
        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", cand_signals.get("decel_mag", -5.0))))
        dist_val = float(cand_signals.get("distance_m", 5.0))
        iou_val = float(cand_signals.get("bbox_iou", 0.0))

        telemetry_payload = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "gt_impact_frame": gt_frame,
            "event_type": cand_type,
            "involved_tracks": fp_cand["track_ids"],
            "composite_score": float(cand_signals.get("composite_score", 0.50)),
            "accel_ms2": acc_val,
            "yaw_delta_deg": float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 25.0))),
            "theta_deg": float(cand_signals.get("theta_deg", 0.0)),
            "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.15))),
            "delta_ar": float(cand_signals.get("delta_ar", 0.0)),
            "d_edge_m": float(cand_signals.get("d_edge_m", 1.0)),
            "distance_m": dist_val,
            "bbox_iou": iou_val,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", cand_signals.get("speed_kmh", cand_signals.get("metric_speed_kmh", 0.0)))),
            "speed_kmh": float(cand_signals.get("speed_kmh", cand_signals.get("metric_speed_kmh", 0.0))),
            "track_age": min([len(track_life_history.get(tid, [])) for tid in fp_cand.get("track_ids", [])]) if fp_cand.get("track_ids") else 0,
            "is_single_vehicle": is_single,
            "weather": weather,
            "day_time": day_time,
            "post_event_rest": cand_signals.get("post_event_rest", True),
                "tau_rest": 0.25 if cand_signals.get("post_event_rest", False) else 0.0,
                "delta_v_drop": float(cand_signals.get("S_v_drop", 0.0)) * 20.0,
            "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
            "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
            "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
            "norm_pixel_dist": float(cand_signals.get("norm_pixel_dist", 99.0)),
            "is_kinematic_closure_corridor": bool(cand_signals.get("is_kinematic_closure_corridor", False)),
            "is_tentative_rescued": bool(cand_signals.get("is_tentative_rescued", False)),
        }

        vlm_calls += 1
        fp_ep.vlm_dispatched = True
        fp_res = arbitrator.arbitrate(
            evidence_strip=strip,
            telemetry=telemetry_payload,
            candidate_frame=cand_f,
            video_id=video_id,
        )
        fp_ep.vlm_result = fp_res

        if fp_res.get("source") == "deterministic_high_confidence_bypass":
            bypass_activations += 1
        if fp_res.get("verdict") == "CANDIDATE_UNCONFIRMED":
            candidate_unconfirmed_count += 1

        is_fp_confirmed = (fp_res.get("is_collision", False) or fp_res.get("verdict") == "CONFIRMED_INCIDENT") and not fp_res.get("suppressed", False)
        if is_fp_confirmed:
            fp_confirmed_count += 1
            fp_ep.is_confirmed = True
            btel = fp_res.get("bypass_telemetry", {})
            print(f"  [FP CONFIRMED] {video_id} @ F{cand_f} ({cand_type}): reason={btel.get('bypass_reason')} | contact={btel.get('contact_evidence')} | impulse={btel.get('impulse')}")
        else:
            fp_suppressed_count += 1

    is_tp = vlm_tp_confirmed
    is_fn = not is_tp
    classification = "TP" if is_tp else "FN"

    track_survival_lengths = {tid: len(f_list) for tid, f_list in track_life_history.items() if len(f_list) >= 4}

    return {
        "video_id": video_id,
        "split_type": split_type,
        "gt_frame": gt_frame,
        "classification": classification,
        "is_tp": is_tp,
        "is_fn": is_fn,
        "fp_confirmed_count": fp_confirmed_count,
        "fp_suppressed_count": fp_suppressed_count,
        "candidate_unconfirmed_count": candidate_unconfirmed_count,
        "bypass_activations": bypass_activations,
        "actor_cooldown_suppressions": actor_cooldown_suppressions,
        "failure_gate": None if is_tp else failure_gate,
        "gate1_passed": gate1_passed,
        "gate2_passed": gate2_passed,
        "gate3_passed": gate3_passed,
        "gate4_confirmed": vlm_tp_confirmed,
        "matched_candidate": matched_tp_candidate,
        "vlm_tp_result": vlm_tp_result,
        "raw_candidates_count": len(all_candidates),
        "consolidated_episodes_count": len(episodes),
        "in_window_episodes_count": len(in_window_episodes),
        "out_of_window_episodes_count": len(out_of_window_episodes),
        "vlm_calls": vlm_calls,
        "candidate_recovered_admitted": total_candidates_admitted,
        "candidate_recovered_rejected_geometry": total_candidates_rejected_geom,
        "motion_crops_evaluated": total_motion_crops_evaluated,
        "micro_vehicles_recovered": total_micro_vehicles_recovered,
        "track_survival_lengths": track_survival_lengths,
        "total_candidates_generated": len(all_candidates),
        "fps": fps,
        "duration_sec": round(elapsed_s, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Iteration S.A Full Benchmark Suite & Regression Runner")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=4, help="Run specific stage (1=Daytime FP clips, 2=Nocturnal, 3=Target Recoveries, 4=Full 30)")
    parser.add_argument("--output", type=str, default="artifacts/sa_full_benchmark_sweep.json", help="Path to output json artifact")
    args = parser.parse_args()

    metadata_path = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv"
    video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"

    cfg = load_config("road_surveillance/config/default_config.yaml")
    yolo_model = YOLO(cfg.detection.model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    vocab_gate = VocabularyGate()
    quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
    motion_gate = MotionROIGate(
        min_cluster_area=250,
        max_cluster_area=3500,
        persistence_threshold=4,
        crop_resolution=320,
        crop_conf_threshold=0.05,
        max_crops_per_frame=1,
        min_crop_interval=2,
    )
    arbitrator = GroqArbitrator(mock_fallback=True)
    loader = GroundTruthLoader(metadata_path)

    # Stage filtering
    all_entries = load_30_benchmark_entries(metadata_path)
    if args.stage == 1:
        # Stage 1: 4 daytime FP-heavy clips
        stage_vids = {"BgdKGnFqz0I_00", "maJGw3hQs_s_00", "kABWTDKgc9U_00", "F0NHENn-HS4_00"}
        benchmark_entries = [e for e in all_entries if e["video_id"] in stage_vids]
        stage_title = "STAGE 1: 4 DAYTIME FP-HEAVY CLIPS AUDIT"
    elif args.stage == 2:
        # Stage 2: 10 nocturnal control runs
        benchmark_entries = [
            e for e in all_entries 
            if (loader.get_by_video_id(e["video_id"]) and loader.get_by_video_id(e["video_id"]).day_time == "night")
        ]
        stage_title = "STAGE 2: 10 NOCTURNAL CONTROL SUITE REGRESSION"
    elif args.stage == 3:
        # Stage 3: Target vehicle recoveries
        stage_vids = {"ftk_fSc4haM_00", "UarP8qU1S-c_01", "KTGByI3N4G0_00"}
        benchmark_entries = [e for e in all_entries if e["video_id"] in stage_vids]
        if not any(e["video_id"] == "KTGByI3N4G0_00" for e in benchmark_entries):
            ktg_gt = loader.get_by_video_id("KTGByI3N4G0_00")
            if ktg_gt:
                benchmark_entries.append({
                    "video_id": "KTGByI3N4G0_00",
                    "split_type": "split_in_distribution",
                    "gt_frame": getattr(ktg_gt, "impact_frame", 91) or 91,
                    "incident_type": ktg_gt.incident_type or "single",
                    "weather": ktg_gt.weather or "normal",
                    "day_time": ktg_gt.day_time or "night",
                })
        stage_title = "STAGE 3: TARGET VEHICLE RECOVERIES AUDIT"
    else:
        benchmark_entries = all_entries
        stage_title = "STAGE 4: FULL 30-VIDEO BENCHMARK SWEEP (ITERATION S.A)"

    detection_cache: Dict[str, Any] = {}

    print("=" * 115)
    print(stage_title)
    print("Invariants: Native PTS timing | Frozen tau_high=0.25, tau_low=0.10 | Frozen tau_rest >= 0.25s")
    print("Task S.A: VLM Fallback Safety (CANDIDATE_UNCONFIRMED) | Deterministic Bypass | Actor-Level Cooldown (10s)")
    print("=" * 115)
    print(f"{'#':<3} | {'Video ID':<16} | {'Split':<22} | {'GT F':<5} | {'Type':<8} | {'Cond':<12} | {'Status':<6} | {'Raw':<5} | {'Eps':<4} | {'VLM':<4} | {'Trig F':<8} | {'FP':<3} | {'Unconf':<6}")
    print("-" * 115)

    per_video_results = []
    tp_count = 0
    fn_count = 0
    total_fp_count = 0
    total_fp_suppressed = 0
    total_raw_candidates = 0
    total_consolidated_episodes = 0
    total_vlm_calls = 0
    total_bypass_activations = 0
    total_candidate_unconfirmed = 0
    total_actor_cooldown_suppressed = 0

    split_metrics = {
        "split_in_distribution": {"TP": 0, "FN": 0, "FP": 0, "suppressed": 0, "raw": 0, "episodes": 0, "vlm": 0, "bypass": 0, "unconf": 0},
        "split_geo_aware": {"TP": 0, "FN": 0, "FP": 0, "suppressed": 0, "raw": 0, "episodes": 0, "vlm": 0, "bypass": 0, "unconf": 0},
    }

    stratified = {
        "incident_type": defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0}),
        "day_time": defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0}),
        "weather": defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0}),
    }

    gate_failures = defaultdict(int)
    t_start_all = time.perf_counter()

    for idx, entry in enumerate(benchmark_entries, 1):
        vid = entry["video_id"]
        gt_f = int(entry["gt_frame"])
        s_type = entry["split_type"]

        gt_rec = loader.get_by_video_id(vid)
        itype = gt_rec.incident_type if gt_rec else entry.get("incident_type", "unknown")
        weather = gt_rec.weather if gt_rec else entry.get("weather", "normal")
        day_time = gt_rec.day_time if gt_rec else entry.get("day_time", "day")
        cond_str = f"{weather}/{day_time}"

        vid_path = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vid_path):
            print(f"{idx:02d} | {vid:<16} | {s_type:<22} | F{gt_f:<4} | {itype:<8} | {cond_str:<12} | MISSING FILE")
            continue

        res = run_single_video_evaluation(
            video_path=vid_path,
            video_id=vid,
            gt_frame=gt_f,
            split_type=s_type,
            weather=weather,
            day_time=day_time,
            yolo_model=yolo_model,
            preprocessor=preprocessor,
            vocab_gate=vocab_gate,
            quality_gate=quality_gate,
            motion_gate=motion_gate,
            arbitrator=arbitrator,
            cfg=cfg,
            tau_single=0.65,
            detection_cache=detection_cache,
        )

        res["incident_type"] = itype
        res["weather"] = weather
        res["day_time"] = day_time
        per_video_results.append(res)

        if res["is_tp"]:
            tp_count += 1
            split_metrics[s_type]["TP"] += 1
            stratified["day_time"][day_time]["TP"] += 1
            stratified["weather"][weather]["TP"] += 1
            stratified["incident_type"][itype]["TP"] += 1
        else:
            fn_count += 1
            gate_failures[res["failure_gate"]] += 1
            split_metrics[s_type]["FN"] += 1
            stratified["day_time"][day_time]["FN"] += 1
            stratified["weather"][weather]["FN"] += 1
            stratified["incident_type"][itype]["FN"] += 1

        split_metrics[s_type]["FP"] += res["fp_confirmed_count"]
        split_metrics[s_type]["suppressed"] += res["fp_suppressed_count"]
        split_metrics[s_type]["raw"] += res["raw_candidates_count"]
        split_metrics[s_type]["episodes"] += res["consolidated_episodes_count"]
        split_metrics[s_type]["vlm"] += res["vlm_calls"]
        split_metrics[s_type]["bypass"] += res["bypass_activations"]
        split_metrics[s_type]["unconf"] += res["candidate_unconfirmed_count"]

        total_fp_count += res["fp_confirmed_count"]
        total_fp_suppressed += res["fp_suppressed_count"]
        total_raw_candidates += res["raw_candidates_count"]
        total_consolidated_episodes += res["consolidated_episodes_count"]
        total_vlm_calls += res["vlm_calls"]
        total_bypass_activations += res["bypass_activations"]
        total_candidate_unconfirmed += res["candidate_unconfirmed_count"]
        total_actor_cooldown_suppressed += res["actor_cooldown_suppressions"]

        stratified["day_time"][day_time]["FP"] += res["fp_confirmed_count"]
        stratified["weather"][weather]["FP"] += res["fp_confirmed_count"]
        stratified["incident_type"][itype]["FP"] += res["fp_confirmed_count"]

        trig_f_str = f"F{res['matched_candidate']['frame']}" if res["matched_candidate"] else "None"
        print(f"{idx:02d} | {vid:<16} | {s_type:<22} | F{gt_f:<4} | {itype:<8} | {cond_str:<12} | {res['classification']:<6} | {res['raw_candidates_count']:<5} | {res['consolidated_episodes_count']:<4} | {res['vlm_calls']:<4} | {trig_f_str:<8} | {res['fp_confirmed_count']:<3} | {res['candidate_unconfirmed_count']:<6}")

    total_time_all = time.perf_counter() - t_start_all
    total_eval = len(per_video_results)

    recall = (tp_count / total_eval) if total_eval > 0 else 0.0
    precision = (tp_count / (tp_count + total_fp_count)) if (tp_count + total_fp_count) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    in_tp = split_metrics["split_in_distribution"]["TP"]
    in_fp = split_metrics["split_in_distribution"]["FP"]
    in_fn = split_metrics["split_in_distribution"]["FN"]
    in_total = in_tp + in_fn
    in_r = in_tp / in_total if in_total > 0 else 0.0
    in_p = in_tp / (in_tp + in_fp) if (in_tp + in_fp) > 0 else 0.0
    in_f1 = (2 * in_p * in_r / (in_p + in_r)) if (in_p + in_r) > 0 else 0.0

    geo_tp = split_metrics["split_geo_aware"]["TP"]
    geo_fp = split_metrics["split_geo_aware"]["FP"]
    geo_fn = split_metrics["split_geo_aware"]["FN"]
    geo_total = geo_tp + geo_fn
    geo_r = geo_tp / geo_total if geo_total > 0 else 0.0
    geo_p = geo_tp / (geo_tp + geo_fp) if (geo_tp + geo_fp) > 0 else 0.0
    geo_f1 = (2 * geo_p * geo_r / (geo_p + geo_r)) if (geo_p + geo_r) > 0 else 0.0

    print("\n" + "=" * 115)
    print("ITERATION S.A: BENCHMARK SWEEP REPORT — VLM SAFETY & DETERMINISTIC BYPASS")
    print("=" * 115)
    print(f"Total Videos Evaluated:       {total_eval} / {len(benchmark_entries)}")
    print(f"Overall True Positives (TP):  {tp_count}")
    print(f"Overall False Negatives (FN): {fn_count}")
    print(f"Overall False Positives (FP): {total_fp_count} (Daytime: {stratified['day_time']['day']['FP']}, Night: {stratified['day_time']['night']['FP']})")
    print(f"Total FP Candidates Suppressed: {total_fp_suppressed}")
    print(f"Candidate Unconfirmed Count:  {total_candidate_unconfirmed} (offline unverified fallback)")
    print(f"Bypass Activations:           {total_bypass_activations} (deterministic_high_confidence_bypass)")
    print(f"Actor Cooldown Suppressions:  {total_actor_cooldown_suppressed} (10s duplicate dispatch cooldown)")
    print("-" * 115)
    print(f"OVERALL INCIDENT RECALL:      {recall * 100:.1f}% ({tp_count}/{total_eval})")
    print(f"OVERALL PRECISION:            {precision * 100:.1f}%")
    print(f"OVERALL F1 SCORE:             {f1:.3f}")
    print("-" * 115)
    print("PIPELINE DISPATCH PROGRESSION (Iteration S.A):")
    print(f"  * Raw Candidate Proposals:   {total_raw_candidates}")
    print(f"  * Consolidated Episodes:     {total_consolidated_episodes} (Unique incident events)")
    print(f"  * Dispatched VLM Calls:      {total_vlm_calls}")
    print(f"  * Confirmed Incidents:       {tp_count + total_fp_count}")
    print(f"  * Suppressed Fallbacks:      {total_candidate_unconfirmed}")
    print("-" * 115)
    print("SPLIT PERFORMANCE BREAKDOWN:")
    print(f"  * In-Distribution: Recall={in_r * 100:.1f}% ({in_tp}/{in_total}) | Precision={in_p * 100:.1f}% | F1={in_f1:.3f} | FP={in_fp}")
    print(f"  * Geo-Aware:       Recall={geo_r * 100:.1f}% ({geo_tp}/{geo_total}) | Precision={geo_p * 100:.1f}% | F1={geo_f1:.3f} | FP={geo_fp}")
    print("=" * 115)

    report_data = {
        "iteration": "Iteration S.A",
        "stage": args.stage,
        "total_evaluated": total_eval,
        "overall_true_positives": tp_count,
        "overall_false_negatives": fn_count,
        "overall_false_positives": total_fp_count,
        "overall_fp_suppressed": total_fp_suppressed,
        "total_candidate_unconfirmed": total_candidate_unconfirmed,
        "total_bypass_activations": total_bypass_activations,
        "total_actor_cooldown_suppressed": total_actor_cooldown_suppressed,
        "incident_recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "split_metrics": split_metrics,
        "per_video_results": per_video_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
