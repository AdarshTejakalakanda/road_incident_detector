"""CLI Entrypoint to Run the Updated Road Surveillance Pipeline (Iteration N).

Usage Examples:
    # Run on a video file path:
    python run_pipeline.py --video "C:/path/to/video.mp4"

    # Run on a video ID from the dataset archive:
    python run_pipeline.py --video "yzNEliTlFGk_00"

    # Run with custom frame limit and evidence output:
    python run_pipeline.py --video "nAXTthLfgtI_00" --max-frames 200 --save-evidence
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import torch

torch.set_num_threads(8)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
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

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import create_evidence_strip, GroqArbitrator
from modules.episode_consolidator import consolidate_candidates_into_episodes, ActorDispatchTracker
from modules.scene_escalator import (
    SceneEscalator,
    SceneEscalatorDisposition,
    EpisodeStatus,
    SceneEscalationResult,
    CandidateProposal,
    TrackState,
    TrackerContext,
    StreamMetadata,
)
from modules.telemetry_accumulator import TelemetryAccumulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def resolve_video_path(video_input: str) -> str:
    """Resolve file path from direct path or dataset archive."""
    if os.path.isfile(video_input):
        return os.path.abspath(video_input)

    vid_base = os.path.basename(video_input)
    if not vid_base.endswith(".mp4"):
        vid_base = f"{vid_base}.mp4"
    candidates = [
        video_input,
        os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", vid_base),
        os.path.join(os.path.expanduser("~"), "Downloads", "archive", "real_videos", vid_base),
        os.path.join("data", "videos", vid_base),
        os.path.join("real_videos", vid_base),
        os.path.join(".", vid_base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(f"Video '{video_input}' could not be located on disk.")


def compute_union_roi(
    boxes: List[Tuple[float, float, float, float]],
    frame_shape: Tuple[int, int],
    margin_factor: float = 0.5,
    min_size: int = 220,
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

    return (
        int(max(0, np.floor(x1 - pad_w))),
        int(max(0, np.floor(y1 - pad_h))),
        int(min(w_img, np.ceil(x2 + pad_w))),
        int(min(h_img, np.ceil(y2 + pad_h))),
    )


def load_frame_at(video_path: str, target_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_idx - 1))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def run_pipeline(
    video_path: str,
    config_path: str = "road_surveillance/config/default_config.yaml",
    max_frames: Optional[int] = None,
    save_evidence: bool = True,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_path = resolve_video_path(video_path)
    video_id = os.path.splitext(os.path.basename(resolved_path))[0]

    print("\n" + "=" * 80)
    print(f"RUNNING UPDATED ROAD SURVEILLANCE PIPELINE (ITERATION N)")
    print(f"Video: {resolved_path}")
    print(f"ID:    {video_id}")
    print("=" * 80)

    cfg = load_config(config_path)
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

    cap_meta = cv2.VideoCapture(resolved_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    stream_fps = float(cap_meta.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap_meta.release()

    stream_metadata = StreamMetadata(
        fps=stream_fps,
        duration_sec=(total_frames / stream_fps) if total_frames > 0 and stream_fps > 0 else 0.0,
        total_frames=total_frames,
        width=vid_w,
        height=vid_h,
    )
    tracker_context = TrackerContext(fps=stream_fps)
    telemetry_acc = TelemetryAccumulator()

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
            d.tau_single = 0.65
            d.set_roadway_polygon(road_poly)

    tracker = RobustSurveillanceTracker()
    motion_gate.reset()

    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    all_candidates: List[Dict[str, Any]] = []
    crops_evaluated = 0
    micro_recovered = 0
    admitted_candidates = 0

    ingestor = DatasetVideoIngestor(path=resolved_path)
    t0 = time.perf_counter()
    last_f_idx = 0

    print("Processing video stream...")
    for frame in ingestor.stream_video(resolved_path):
        f_idx = frame.frame_idx
        if max_frames and f_idx > max_frames:
            break
        last_f_idx = f_idx
        img = frame.image
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)

        # 1. Primary YOLO 640p Pass
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

        # 2. Motion-Guided ROI Zoom Pass (Patch N.2)
        recovered_micro, m_tel = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )
        raw_dets.extend(recovered_micro)
        crops_evaluated += m_tel["roi_crops_evaluated"]
        micro_recovered += m_tel["recovered_micro_vehicles"]

        # 3. Detector Quality Gate (Patch N.1 semantic recovery)
        admitted, qtel = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )
        admitted_candidates += qtel.get("candidate_recovered_admitted", 0)

        # 4. Robust Tracker
        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

        frame_tracks_cache[f_idx] = {
            tid: (
                tr.current_bbox_xywh[0],
                tr.current_bbox_xywh[1],
                tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2],
                tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3],
            )
            for tid, tr in active_tracks.items()
        }

        # 5. Kinematics & Events
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
        telemetry_acc.record_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            timestamp_s=timestamp_s,
            active_tracks=active_tracks,
            kinematics_map=kinematics,
        )
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
            print(f"  [ALERT F{f_idx:03d} | {timestamp_s:.2f}s] {etype.upper()} (conf={c.confidence:.2f}, tracks={c.track_ids})")

    elapsed_s = time.perf_counter() - t0
    fps = round(last_f_idx / elapsed_s, 1) if elapsed_s > 0 else 0.0

    # 6. Episode Consolidation & Video-VLM Arbitration (Iteration P.3)
    episodes = consolidate_candidates_into_episodes(
        raw_candidates=all_candidates,
        video_id=video_id,
        cooldown_seconds=10.0,
        max_gap_frames=60,
    )

    if stream_metadata.duration_sec <= 0.0 or (max_frames and last_f_idx < stream_metadata.total_frames):
        stream_metadata.duration_sec = last_f_idx / stream_metadata.fps

    # --- PHASE 2 SCENE ESCALATOR INTEGRATION ---
    print(f"\nEvaluating {len(episodes)} consolidated episodes with SceneEscalator (tau_aftermath=0.50)...")
    escalated_episodes = []
    quarantined_records = []
    all_aftermath_audit_records = []
    scene_escalator = SceneEscalator(tau_aftermath=0.50)

    # Reconcile stream end timestamp with synthetic timebase
    stream_duration_s = float(total_frames / stream_fps) if (total_frames > 0 and stream_fps > 0) else 0.0
    if max_frames and last_f_idx < total_frames and stream_fps > 0:
        t_stream_end = float(last_f_idx / stream_fps)
    else:
        t_stream_end = float(total_frames / stream_fps) if stream_fps > 0 else stream_duration_s

    for ep in episodes:
        t_anchor = float(ep.peak_candidate["timestamp_s"])
        delta_t_avail = t_stream_end - t_anchor

        if delta_t_avail < 0.75:
            res = SceneEscalationResult(
                disposition=SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW,
                score=None,
                window_state="INSUFFICIENT_AFTERMATH_WINDOW",
                telemetry={"delta_t_avail": delta_t_avail},
            )
        else:
            t_end = min(t_anchor + 1.50, t_stream_end)
            window_history = telemetry_acc.get_window(t_start=t_anchor, t_end=t_end)

            res = scene_escalator.evaluate_candidate(
                candidate=ep.peak_candidate,
                track_history=window_history,
                fps=stream_fps,
            )
            res.window_state = (
                "COMPLETE_AFTERMATH_WINDOW" if delta_t_avail >= 1.50
                else "PARTIAL_AFTERMATH_WINDOW"
            )

        ep.aftermath_result = res

        # Safely extract actual fields from SceneEscalationResult
        raw_telemetry = getattr(res, "features", getattr(res, "telemetry", {}))
        normalized_channels = getattr(res, "channel_scores", getattr(res, "channels", {}))
        raw_window_state = getattr(res, "window_state", "UNKNOWN")
        raw_disposition = getattr(res, "disposition", "UNKNOWN")
        res_score = getattr(res, "score", getattr(res, "aftermath_score", None))

        unified_audit_entry = {
            "episode_id": ep.episode_id,
            "anchor_frame": ep.peak_candidate["frame"],
            "canonical_type": getattr(res, "canonical_type", "UNKNOWN"),
            "disposition": raw_disposition.value if hasattr(raw_disposition, "value") else str(raw_disposition),
            "score": round(float(res_score), 4) if (res_score is not None and np.isfinite(res_score)) else None,
            "window_state": raw_window_state.value if hasattr(raw_window_state, "value") else str(raw_window_state),
            "telemetry": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (None if isinstance(v, float) and not np.isfinite(v) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in raw_telemetry.items()},
            "channels": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (None if isinstance(v, float) and not np.isfinite(v) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in normalized_channels.items()},
            "safeguards": getattr(res, "safeguards", {}),
        }
        all_aftermath_audit_records.append(unified_audit_entry)

        # Backward-compatible suppression record format
        suppression_entry = {
            "episode_id": ep.episode_id,
            "anchor_frame": ep.peak_candidate["frame"],
            "disposition": res.disposition.value if hasattr(res.disposition, "value") else str(res.disposition),
            "score": res.score,
            "window_state": res.window_state,
            "telemetry": res.telemetry,
            "channels": res.channels,
        }

        if res.disposition in {
            SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT,
            SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW,
        }:
            escalated_episodes.append(ep)
            print(f"  [ESCALATED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> {res.disposition.value} (score={res.score}, type={res.canonical_type})")
        else:
            ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
            quarantined_records.append(suppression_entry)
            print(f"  [SUPPRESSED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (score={res.score}, type={res.canonical_type})")

    # Write unified run artifact containing 100% of evaluated episodes
    os.makedirs("artifacts", exist_ok=True)
    audit_path_all = f"artifacts/aftermath_evaluation_audit_{video_id}.json"
    with open(audit_path_all, "w", encoding="utf-8") as f:
        json.dump(all_aftermath_audit_records, f, indent=2)
    print(f"Unified aftermath audit saved: {audit_path_all} ({len(all_aftermath_audit_records)} evaluated records)")

    # Preserve backward compatibility: write suppression audit as well
    audit_path_suppressed = f"artifacts/aftermath_suppression_audit_{video_id}.json"
    with open(audit_path_suppressed, "w", encoding="utf-8") as f:
        json.dump(quarantined_records, f, indent=2)
    print(f"Quarantine audit saved: {audit_path_suppressed} ({len(quarantined_records)} suppressed records)")

    # Pass ONLY escalated episodes to downstream Gate 4 arbitration
    episodes = escalated_episodes

    print(f"\nCompleted {last_f_idx} frames in {elapsed_s:.2f}s ({fps} FPS).")
    print(f"Progression: {len(all_candidates)} Raw Proposals -> {len(episodes)} Escalated Episodes.")
    print(f"Arbitrating {len(episodes)} episodes via Video-VLM Arbitration Layer...")

    arbitrated_incidents = []
    dispatch_tracker = ActorDispatchTracker(cooldown_seconds=10.0)
    os.makedirs("artifacts", exist_ok=True)

    for idx, ep in enumerate(episodes, 1):
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_t = cand.get("timestamp_s", cand_f / 30.0)
        t_app = max(1, cand_f - 8)
        t_imp = cand_f
        t_post = min(last_f_idx, cand_f + 8)

        # Task S.A.3: Actor-level dispatch cooldown check
        should_suppress, cd_log = dispatch_tracker.should_suppress_dispatch(cand.get("track_ids", []), cand_t)
        if should_suppress:
            ep.dispatch_suppressed = True
            ep.suppression_reason = "actor_cooldown_active"
            ep.cooldown_telemetry = cd_log
            print(f"  [{idx}/{len(episodes)}] Episode {ep.episode_id} @ F{cand_f:03d} -> SUPPRESSED (actor_cooldown_active: {cd_log['actor_ids']} dt={cd_log['delta_time_s']}s)")
            continue

        dispatch_tracker.record_dispatch(cand.get("track_ids", []), cand_t)

        img_app = load_frame_at(resolved_path, t_app)
        img_imp = load_frame_at(resolved_path, t_imp)
        img_post = load_frame_at(resolved_path, t_post)

        boxes = []
        for f in [t_app, t_imp, t_post]:
            if f in frame_tracks_cache:
                boxes.extend(frame_tracks_cache[f].values())

        roi = compute_union_roi(boxes, (vid_h, vid_w), margin_factor=0.6, min_size=260)
        rx1, ry1, rx2, ry2 = roi

        strip = create_evidence_strip(
            crop_pre=img_app[ry1:ry2, rx1:rx2],
            crop_impact=img_imp[ry1:ry2, rx1:rx2],
            crop_post=img_post[ry1:ry2, rx1:rx2],
            target_h=336,
            labels=(f"1: APPROACH (F{t_app})", f"2: INCIDENT (F{t_imp})", f"3: POST (F{t_post})"),
        )

        if save_evidence:
            strip_path = f"artifacts/{video_id}_ep_{idx:02d}_f{cand_f:03d}_evidence.jpg"
            cv2.imwrite(strip_path, strip)

        cand_signals = cand.get("signals", {})
        cand_type = cand["event_type"]
        is_single = (cand_type in ("single_vehicle_anomaly", "single"))
        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -4.0)))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        iou_val = float(cand_signals.get("bbox_iou", 0.10))

        telemetry = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "event_type": cand_type,
            "involved_tracks": cand["track_ids"],
            "composite_score": float(cand_signals.get("composite_score", 0.65)),
            "accel_ms2": acc_val,
            "yaw_delta_deg": float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 30.0))),
            "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
            "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
            "distance_m": dist_val,
            "bbox_iou": iou_val,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", 30.0)),
            "speed_kmh": float(cand_signals.get("metric_speed_kmh", 5.0)),
            "track_age": 10,
            "is_single_vehicle": is_single,
            "post_event_rest": cand_signals.get("post_event_rest", True),
            "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
            "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
            "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
        }

        res = arbitrator.arbitrate(
            evidence_strip=strip,
            telemetry=telemetry,
            candidate_frame=cand_f,
            video_id=video_id,
        )

        is_confirmed = (res.get("is_collision", False) or res.get("verdict") == "CONFIRMED_INCIDENT") and not res.get("suppressed", False)
        verdict_str = res.get("verdict", "UNKNOWN")
        if is_confirmed:
            status_tag = f"CONFIRMED ({res.get('source', '')})"
        elif verdict_str == "CANDIDATE_UNCONFIRMED":
            status_tag = f"UNCONFIRMED FALLBACK ({res.get('suppression_reason', '')})"
        else:
            status_tag = "REJECTED (FALSE ALARM)"

        print(f"  [{idx}/{len(episodes)}] Episode {ep.episode_id} @ F{cand_f:03d} -> {status_tag} (conf={res.get('confidence', 0.0):.2f}): {res.get('rationale', '')}")

        if is_confirmed:
            arbitrated_incidents.append({
                "episode": ep,
                "candidate": cand,
                "arbitration": res,
            })

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Video:                   {resolved_path}")
    print(f"Frames Processed:        {last_f_idx}")
    print(f"Throughput:              {fps} FPS")
    print(f"Micro-Crops Evaluated:   {crops_evaluated}")
    print(f"Micro-Vehicles Found:    {micro_recovered}")
    print(f"Audited Detections:      {admitted_candidates}")
    print(f"Raw Proposals:           {len(all_candidates)}")
    print(f"Consolidated Episodes:   {len(episodes)}")
    print(f"Confirmed Incidents:     {len(arbitrated_incidents)}")
    for inc in arbitrated_incidents:
        c = inc["candidate"]
        a = inc["arbitration"]
        print(f"  * Frame {c['frame']} ({c['timestamp_s']}s) | {a.get('collision_type', c['event_type'])} | Conf: {a.get('confidence', 0.0):.2f}")
        print(f"    Rationale: {a.get('rationale', '')}")
    acc_mem = telemetry_acc.get_total_memory_usage()
    print(f"TelemetryAccumulator Peak Memory: {acc_mem['total_bytes']:,} bytes ({acc_mem['total_mib']} MiB) | Capacity: {acc_mem['allocated_capacity']:,} rows | Snapshots: {acc_mem['recorded_snapshots']:,} | Buffer: {acc_mem['buffer_bytes']:,} bytes | Reallocations: {acc_mem['reallocations']}")
    print("=" * 80)

    result_summary = {
        "video_id": video_id,
        "video_path": resolved_path,
        "frames_processed": last_f_idx,
        "fps": fps,
        "candidates_count": len(all_candidates),
        "confirmed_count": len(arbitrated_incidents),
        "crops_evaluated": crops_evaluated,
        "micro_recovered": micro_recovered,
        "admitted_recovered": admitted_candidates,
        "confirmed_incidents": arbitrated_incidents,
        "telemetry_accumulator_memory": acc_mem,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)
        print(f"Saved results to: {output_json}")

    return result_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Updated Road Surveillance Pipeline (Iteration N)")
    parser.add_argument("--video", "--video-path", "-v", type=str, required=True, help="Path to input video file or dataset video ID")
    parser.add_argument("--config", "--config-path", "-c", type=str, default="road_surveillance/config/default_config.yaml", help="Path to configuration yaml")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-evidence", action="store_true", help="Do not save evidence filmstrip image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        config_path=args.config,
        max_frames=args.max_frames,
        save_evidence=not args.no_evidence,
        output_json=args.output,
    )
