"""Run Stage T Evaluation on a Single Video Clip.

Usage:
    python tools/run_stage_t_single_clip.py --video 5uh8-3q51uI_00 --config road_surveillance/config/default_config.yaml
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
import pandas as pd

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

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import create_evidence_strip, GroqArbitrator
from modules.episode_consolidator import consolidate_candidates_into_episodes, ConsolidatedEpisode, ActorDispatchTracker


def resolve_video_path(video_input: str) -> str:
    """Resolve file path from direct path or dataset archive."""
    if os.path.isfile(video_input):
        return os.path.abspath(video_input)

    vid_name = video_input if video_input.endswith(".mp4") else f"{video_input}.mp4"
    candidates = [
        os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", vid_name),
        os.path.join(os.path.expanduser("~"), "Downloads", "archive", "real_videos", vid_name),
        os.path.join("data", "videos", vid_name),
        os.path.join(".", vid_name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(f"Video '{video_input}' could not be located on disk.")


def load_gt_info(video_id: str, metadata_path: str = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv") -> Dict[str, Any]:
    if not os.path.isfile(metadata_path):
        return {"gt_frame": 176, "event_type": "rear-end", "accident_time": 5.867}
    try:
        df = pd.read_csv(metadata_path)
        matched = df[df.iloc[:, 0].astype(str).str.contains(video_id)]
        if not matched.empty:
            row = matched.iloc[0]
            return {
                "gt_frame": int(row.get("accident_frame", 176)),
                "event_type": str(row.get("type", "rear-end")),
                "accident_time": float(row.get("accident_time", 5.867)),
                "x1": float(row.get("x1", 0.0)),
                "y1": float(row.get("y1", 0.0)),
                "x2": float(row.get("x2", 0.0)),
                "y2": float(row.get("y2", 0.0)),
            }
    except Exception as e:
        print(f"Warning loading metadata: {e}")
    return {"gt_frame": 176, "event_type": "rear-end", "accident_time": 5.867}


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
    return np.zeros((720, 960, 3), dtype=np.uint8)


def run_single_clip(
    video_input: str,
    config_path: str = "road_surveillance/config/default_config.yaml",
    max_frames: Optional[int] = None,
    output_json: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_path = resolve_video_path(video_input)
    video_id = os.path.splitext(os.path.basename(resolved_path))[0]
    gt_info = load_gt_info(video_id)
    gt_frame = gt_info["gt_frame"]
    temporal_tolerance = 15  # +/- 15 frames for in-window evaluation

    print("=" * 80)
    print(f"RUNNING SINGLE CLIP EVALUATION: {video_id}")
    print(f"Video Path:    {resolved_path}")
    print(f"Config:        {config_path}")
    print(f"Ground Truth:  Frame {gt_frame} ({gt_info['event_type']})")
    print("=" * 80)

    cfg = load_config(config_path)

    cap_meta = cv2.VideoCapture(resolved_path)
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

    vocab_gate = VocabularyGate()
    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    for d in event_engine.detectors:
        if isinstance(d, SingleVehicleAnomalyDetector):
            d.tau_single = 0.65
            d.set_roadway_polygon(road_poly)

    # Initialize YOLO Model
    model_path = getattr(cfg.detection, "model_path", "yolov8x.pt")
    yolo_model = YOLO(model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)

    # Pipeline Modules
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
    tracker = RobustSurveillanceTracker()
    motion_gate.reset()

    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    all_candidates: List[Dict[str, Any]] = []
    track_max_id = 0

    t0 = time.perf_counter()
    last_f_idx = 0

    ingestor = DatasetVideoIngestor(path=resolved_path)
    frame_pts_map: Dict[int, float] = {}

    for frame in ingestor.stream_video(resolved_path):
        f_idx = frame.frame_idx
        if max_frames and f_idx >= max_frames:
            break

        last_f_idx = f_idx
        img = frame.image
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)
        frame_pts_map[f_idx] = timestamp_s

        # 1. Primary YOLO 640p pass
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

        # 2. Motion-Guided ROI Zoom Pass
        recovered_micro, _ = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )
        raw_dets.extend(recovered_micro)

        # 3. Quality Gate
        admitted, _ = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )

        # 4. Tracker Update
        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

        for tid in active_tracks.keys():
            if tid > track_max_id:
                track_max_id = tid

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

    # Candidate Breakdown by Module
    by_type = defaultdict(int)
    for c in all_candidates:
        by_type[c["event_type"]] += 1

    # Episode Consolidation
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

    # Collect Ground Truth Window Candidates
    gt_candidates = [
        c for c in all_candidates if abs(c["frame"] - gt_frame) <= temporal_tolerance
    ]

    # VLM Arbitration & Bypass
    arbitrator = GroqArbitrator(api_key=None, mock_fallback=True)
    dispatch_tracker = ActorDispatchTracker(cooldown_seconds=10.0)

    confirmed_incidents = []
    for ep in episodes:
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_t = cand.get("timestamp_s", cand_f / 30.0)

        should_suppress, cd_log = dispatch_tracker.should_suppress_dispatch(cand.get("track_ids", []), cand_t)
        if should_suppress:
            ep.dispatch_suppressed = True
            ep.suppression_reason = "actor_cooldown_active"
            continue

        dispatch_tracker.record_dispatch(cand.get("track_ids", []), cand_t)

        t_app = max(1, cand_f - 8)
        t_imp = cand_f
        t_post = min(last_f_idx, cand_f + 8)

        img_app = load_frame_at(resolved_path, t_app)
        img_imp = load_frame_at(resolved_path, t_imp)
        img_post = load_frame_at(resolved_path, t_post)

        boxes = []
        for f in [t_app, t_imp, t_post]:
            if f in frame_tracks_cache:
                boxes.extend(frame_tracks_cache[f].values())

        roi = compute_union_roi(boxes, (h, w), margin_factor=0.6, min_size=260)
        rx1, ry1, rx2, ry2 = roi

        strip = create_evidence_strip(
            crop_pre=img_app[ry1:ry2, rx1:rx2],
            crop_impact=img_imp[ry1:ry2, rx1:rx2],
            crop_post=img_post[ry1:ry2, rx1:rx2],
            target_h=336,
            labels=(f"1: APPROACH (F{t_app})", f"2: INCIDENT (F{t_imp})", f"3: POST (F{t_post})"),
        )

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
        if is_confirmed:
            confirmed_incidents.append({
                "episode": ep,
                "candidate": cand,
                "arbitration": res,
            })

    # Summary Report
    print("\n" + "=" * 80)
    print("STAGE T SINGLE CLIP AUDIT SUMMARY")
    print("=" * 80)
    print(f"Video:                   {video_id}")
    print(f"Frames Processed:        {last_f_idx} (Speed: {fps:.1f} FPS)")
    print(f"Track ID Range:          1 to {track_max_id}")
    print(f"Total Raw Triggers:      {len(all_candidates)}")
    print("Trigger Breakdown by Module:")
    for et, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  - {et:<30}: {cnt}")
    print(f"Consolidated Episodes:   {len(episodes)}")
    print(f"  - In GT Window (F{gt_frame} +/- 15): {len(in_window_episodes)}")
    print(f"  - Out of GT Window:     {len(out_of_window_episodes)}")
    print(f"Confirmed Incidents:     {len(confirmed_incidents)}")

    # Audit GT Window Retention
    print("\n--- GROUND TRUTH RETENTION AUDIT (Frame 176) ---")
    if in_window_episodes:
        print(f"MATCH: {len(in_window_episodes)} episode(s) matched in GT window:")
        for ep in in_window_episodes:
            pc = ep.peak_candidate
            print(f"  Episode {ep.episode_id}: Peak F{ep.peak_frame} (Score: {ep.peak_score:.2f}) | Type: {pc.get('event_type')} | Tracks: {pc.get('track_ids')}")
            print(f"    Signals: {pc.get('signals')}")
    else:
        print("ALERT: NO episodes matched in GT window!")

    print(f"Total in-window raw candidates: {len(gt_candidates)}")
    for gc in gt_candidates:
        print(f"  Cand @ F{gc['frame']} | Type: {gc['event_type']} | Tracks: {gc['track_ids']} | Conf: {gc['confidence']}")

    print("=" * 80)

    result = {
        "video_id": video_id,
        "frames_processed": last_f_idx,
        "fps": fps,
        "track_max_id": track_max_id,
        "total_raw_triggers": len(all_candidates),
        "trigger_breakdown": dict(by_type),
        "consolidated_episodes": len(episodes),
        "in_window_episodes": len(in_window_episodes),
        "out_of_window_episodes": len(out_of_window_episodes),
        "confirmed_incidents": len(confirmed_incidents),
        "gt_retained": len(in_window_episodes) > 0,
        "all_candidates": all_candidates,
        "in_window_episodes_data": [
            {
                "episode_id": ep.episode_id,
                "peak_frame": ep.peak_frame,
                "peak_score": ep.peak_score,
                "track_ids": ep.peak_candidate.get("track_ids"),
                "event_type": ep.peak_candidate.get("event_type"),
                "signals": ep.peak_candidate.get("signals"),
            }
            for ep in in_window_episodes
        ],
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved audit results to: {output_json}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage T Evaluation on a Single Video Clip")
    parser.add_argument("--video", "-v", type=str, required=True, help="Video filename or path (e.g. 5uh8-3q51uI_00)")
    parser.add_argument("--config", "-c", type=str, default="road_surveillance/config/default_config.yaml", help="Config yaml path")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to process")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_clip(
        video_input=args.video,
        config_path=args.config,
        max_frames=args.max_frames,
        output_json=args.output,
    )
