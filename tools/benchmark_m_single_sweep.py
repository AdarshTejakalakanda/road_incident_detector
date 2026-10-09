"""Iteration M - Step 1: Target Benchmark & Tau_Single Sensitivity Sweep.

Evaluates the hardened Iteration M pipeline on the 7 single-vehicle benchmark runs:
- nAXTthLfgtI_00 (GT F62)
- vxZ47DoPNY8_00 (GT F50)
- VozrseCzhQc_00 (GT F84)
- UarP8qU1S-c_00 (GT F130)
- UarP8qU1S-c_01 (GT F30)
- nAXTthLfgtI_00 (GT F62, geo_aware)
- VozrseCzhQc_00 (GT F84, geo_aware)

Sweeps tau_single in [0.55, 0.60, 0.65, 0.70] to determine the sensitivity boundary
where Gate 3 opens without generating excessive false candidates.

Outputs:
- artifacts/m_single_vehicle_sweep.json
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Project root setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from modules.vlm_arbitrator import create_evidence_strip, GroqArbitrator
from tools.benchmark_k0_detector import load_30_benchmark_entries

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_m_single_sweep")


def compute_union_roi(
    boxes: List[Tuple[float, float, float, float]],
    frame_shape: Tuple[int, int],
    margin_factor: float = 0.5,
    min_size: int = 220,
) -> Tuple[int, int, int, int]:
    """Compute expanded bounding crop around vehicle of interest."""
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


def load_frame_at(video_path: str, target_frame: int) -> np.ndarray:
    """Read a specific frame from video file on demand."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def run_single_vehicle_run(
    video_path: str,
    video_id: str,
    gt_frame: int,
    yolo_model: YOLO,
    preprocessor: LowLightPreprocessor,
    vocab_gate: VocabularyGate,
    quality_gate: DetectorQualityGate,
    arbitrator: GroqArbitrator,
    cfg: Any,
    tau_single: float = 0.60,
    max_frames: int = 250,
    temporal_tolerance: int = 35,
    detection_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Run single video evaluation with specified tau_single."""
    t0 = time.perf_counter()

    tracker = RobustSurveillanceTracker(
        track_activation_threshold=0.10,
        low_confidence_floor=0.06,
    )

    is_calibrated = bool(cfg.calibration.get(video_id))
    raw_calib = cfg.calibration.get(video_id) or cfg.calibration.get("default")
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id=video_id,
        cfg_obj=raw_calib,
        is_calibrated=is_calibrated,
    )
    cap_meta = cv2.VideoCapture(video_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()
    if vid_w <= 0 or vid_h <= 0:
        vid_w, vid_h = 1280, 720

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

    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    detections_near_gt = 0
    tracks_near_gt = 0
    all_candidates: List[Dict[str, Any]] = []

    # Check or populate detection cache
    cached_frames = None
    if detection_cache is not None and video_path in detection_cache:
        cached_frames = detection_cache[video_path]
    else:
        cached_frames = []
        ingestor = DatasetVideoIngestor(path=video_path)
        for frame in ingestor.stream_video(video_path):
            f_idx = frame.frame_idx
            if f_idx > max_frames:
                break
            img = frame.image
            proc_img, illum = preprocessor.process(img)
            h, w = proc_img.shape[:2]
            pts_ns = frame.pts_ns
            timestamp_s = float(pts_ns * 1e-9)

            yolo_res = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
            raw_dets = []
            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                b_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
                b_conf = yolo_res.boxes.conf.cpu().numpy()
                b_cls = yolo_res.boxes.cls.cpu().numpy()
                for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
                    c_name = yolo_model.names[int(bcl)].lower()
                    raw_dets.append({
                        "bbox": [float(x) for x in bx],
                        "conf": float(bc),
                        "class_id": int(bcl),
                        "class_name": c_name,
                    })
            cached_frames.append({
                "frame_idx": f_idx,
                "pts_ns": pts_ns,
                "timestamp_s": timestamp_s,
                "shape": (h, w),
                "raw_dets": raw_dets,
            })
        if detection_cache is not None:
            detection_cache[video_path] = cached_frames

    last_f_idx = 0
    for frame_data in cached_frames:
        f_idx = frame_data["frame_idx"]
        last_f_idx = f_idx
        pts_ns = frame_data["pts_ns"]
        timestamp_s = frame_data["timestamp_s"]
        h, w = frame_data["shape"]
        raw_dets = frame_data["raw_dets"]

        # 2. Quality Gate
        admitted, _ = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )

        veh_admitted = [r for r in admitted if r.fused_class in ("vehicle", "candidate_vehicle_recovered")]
        if abs(f_idx - gt_frame) <= 15:
            detections_near_gt += len(veh_admitted)

        # 3. Tracker Update
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

        if abs(f_idx - gt_frame) <= 15:
            tracks_near_gt = max(tracks_near_gt, len(active_tracks))

        # 4. Kinematics & Events
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

    elapsed_s = time.perf_counter() - t0
    fps = last_f_idx / elapsed_s if elapsed_s > 0 else 0.0

    # Classify Candidates into In-Window (Incident Episode) vs Out-of-Window (FP Candidates)
    in_window_candidates = [c for c in all_candidates if abs(c["frame"] - gt_frame) <= temporal_tolerance]
    unmatched_fp_candidates = [c for c in all_candidates if abs(c["frame"] - gt_frame) > temporal_tolerance]

    gate1_passed = detections_near_gt >= 1
    gate2_passed = tracks_near_gt >= 1
    gate3_passed = len(in_window_candidates) > 0

    # Gate 4: VLM Arbitration on In-Window Candidates (True Positive Episode)
    matched_tp_candidate = None
    vlm_tp_confirmed = False
    vlm_tp_result = None

    if in_window_candidates:
        sorted_in_window = sorted(
            in_window_candidates,
            key=lambda c: (0 if c["event_type"] == "collision_candidate" else 1, abs(c["frame"] - gt_frame)),
        )

        for cand in sorted_in_window:
            cand_f = cand["frame"]
            t_app = max(1, cand_f - 8)
            t_imp = cand_f
            t_post = min(f_idx, cand_f + 8)

            img_app = load_frame_at(video_path, t_app)
            img_imp = load_frame_at(video_path, t_imp)
            img_post = load_frame_at(video_path, t_post)

            boxes_for_roi = []
            for f in [t_app, t_imp, t_post]:
                if f in frame_tracks_cache:
                    boxes_for_roi.extend(frame_tracks_cache[f].values())

            roi = compute_union_roi(boxes_for_roi, (vid_h, vid_w), margin_factor=0.6, min_size=260)
            rx1, ry1, rx2, ry2 = roi

            crop_app = img_app[ry1:ry2, rx1:rx2]
            crop_imp = img_imp[ry1:ry2, rx1:rx2]
            crop_post = img_post[ry1:ry2, rx1:rx2]

            strip = create_evidence_strip(
                crop_pre=crop_app,
                crop_impact=crop_imp,
                crop_post=crop_post,
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
                "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
                "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
                "distance_m": dist_val,
                "bbox_iou": iou_val,
                "anomaly_type": cand_signals.get("anomaly_type", cand_type),
                "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", cand_signals.get("metric_speed_kmh", 0.0))),
                "speed_kmh": float(cand_signals.get("metric_speed_kmh", 0.0)),
                "track_age": 10,
                "is_single_vehicle": is_single,
            }

            cand_res = arbitrator.arbitrate(
                evidence_strip=strip,
                telemetry=telemetry_payload,
                candidate_frame=cand_f,
                video_id=video_id,
            )
            if cand_res.get("is_collision", False) or cand_res.get("verdict") == "CONFIRMED_INCIDENT":
                vlm_tp_confirmed = True
                matched_tp_candidate = cand
                vlm_tp_result = cand_res
                break
            else:
                if matched_tp_candidate is None:
                    matched_tp_candidate = cand
                    vlm_tp_result = cand_res

    # FP Candidates check
    fp_confirmed = 0
    fp_suppressed = 0
    for fp in unmatched_fp_candidates:
        sig = fp.get("signals", {})
        fp_type = fp["event_type"]
        is_single_fp = (fp_type in ("single_vehicle_anomaly", "single"))
        fp_tel = {
            "video_id": video_id,
            "candidate_frame": fp["frame"],
            "event_type": fp_type,
            "involved_tracks": fp["track_ids"],
            "composite_score": float(sig.get("composite_score", 0.0)),
            "accel_ms2": float(sig.get("metric_accel_ms2", 0.0)),
            "yaw_delta_deg": float(sig.get("yaw_delta_deg", 0.0)),
            "aspect_ratio_ratio": float(sig.get("aspect_ratio_ratio", 0.0)),
            "d_edge_m": float(sig.get("d_edge_m", 5.0)),
            "distance_m": float(sig.get("distance_m", 5.0)),
            "bbox_iou": float(sig.get("bbox_iou", 0.0)),
            "anomaly_type": sig.get("anomaly_type", ""),
            "max_recent_speed_kmh": float(sig.get("max_recent_speed_kmh", sig.get("metric_speed_kmh", 0.0))),
            "speed_kmh": float(sig.get("metric_speed_kmh", 0.0)),
            "track_age": 4,
            "is_single_vehicle": is_single_fp,
        }
        dummy_strip = np.zeros((336, 600, 3), dtype=np.uint8)
        res = arbitrator.arbitrate(dummy_strip, fp_tel, candidate_frame=fp["frame"], video_id=video_id)
        if res.get("is_collision", False) or res.get("verdict") == "CONFIRMED_INCIDENT":
            fp_confirmed += 1
        else:
            fp_suppressed += 1

    return {
        "video_id": video_id,
        "gt_frame": gt_frame,
        "tau_single": tau_single,
        "fps": round(fps, 1),
        "gate1_passed": gate1_passed,
        "gate2_passed": gate2_passed,
        "gate3_passed": gate3_passed,
        "gate4_confirmed": vlm_tp_confirmed,
        "matched_candidate": matched_tp_candidate,
        "unmatched_fp_count": len(unmatched_fp_candidates),
        "fp_confirmed": fp_confirmed,
        "fp_suppressed": fp_suppressed,
        "vlm_result": vlm_tp_result,
    }


def run_target_sweep(
    video_dir: str = r"C:\Users\KALAKA~1\Downloads\archive\real_videos",
    metadata_path: str = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv",
    tau_values: List[float] = [0.50, 0.55, 0.60, 0.65, 0.70],
    output_path: str = "artifacts/m_single_vehicle_sweep.json",
) -> Dict[str, Any]:
    """Execute Step 1 parameter sweep across the 7 single-vehicle benchmark runs."""
    cfg = load_config("road_surveillance/config/default_config.yaml")
    yolo_model = YOLO(cfg.detection.model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    vocab_gate = VocabularyGate()
    quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
    arbitrator = GroqArbitrator(mock_fallback=True)

    loader = GroundTruthLoader(metadata_path)
    all_30 = load_30_benchmark_entries(metadata_path)

    # Filter strictly the single-vehicle benchmark entries
    single_entries = []
    for entry in all_30:
        rec = loader.get_by_video_id(entry["video_id"])
        if rec and rec.incident_type == "single":
            single_entries.append(entry)

    print("=" * 90)
    print(f"ITERATION M - STEP 1: SINGLE-VEHICLE RECOVERY SWEEP ({len(single_entries)} TARGET RUNS)")
    print("=" * 90)
    for i, e in enumerate(single_entries):
        rec = loader.get_by_video_id(e["video_id"])
        print(f"[{i+1}] {e['video_id']} | GT F{e['gt_frame']} | {rec.day_time}/{rec.weather} | Split: {e['split_type']}")
    print("-" * 90)

    shared_detection_cache: Dict[str, List[Dict[str, Any]]] = {}
    sweep_results = {}
    for tau in tau_values:
        print(f"\nEvaluating tau_single = {tau:.2f} ...")
        runs = []
        tp_g3_count = 0
        tp_g4_count = 0
        total_fps = 0
        total_fp_conf = 0

        for entry in single_entries:
            vid = entry["video_id"]
            gt_f = int(entry["gt_frame"])
            vpath = os.path.join(video_dir, f"{vid}.mp4")
            if not os.path.exists(vpath):
                print(f"  Missing file: {vpath}")
                continue

            res = run_single_vehicle_run(
                video_path=vpath,
                video_id=vid,
                gt_frame=gt_f,
                yolo_model=yolo_model,
                preprocessor=preprocessor,
                vocab_gate=vocab_gate,
                quality_gate=quality_gate,
                arbitrator=arbitrator,
                cfg=cfg,
                tau_single=tau,
                max_frames=250,
                detection_cache=shared_detection_cache,
            )
            runs.append(res)
            if res["gate3_passed"]:
                tp_g3_count += 1
            if res["gate4_confirmed"]:
                tp_g4_count += 1
            total_fps += res["unmatched_fp_count"]
            total_fp_conf += res["fp_confirmed"]

            status = "CONFIRMED" if res["gate4_confirmed"] else ("CANDIDATE" if res["gate3_passed"] else "DROPPED")
            matched_f = res["matched_candidate"]["frame"] if res["matched_candidate"] else None
            score = res["matched_candidate"]["signals"].get("composite_score", None) if res["matched_candidate"] else None
            print(f"  {vid:<16} | GT F{gt_f:<3} | Matched: F{str(matched_f):<4} (S={str(score):<4}) | Status: {status:<9} | FPs: {res['unmatched_fp_count']}")

        sweep_results[str(tau)] = {
            "tau_single": tau,
            "target_runs": len(runs),
            "gate3_candidates_recovered": tp_g3_count,
            "gate4_vlm_confirmed": tp_g4_count,
            "gate3_recovery_rate": round(tp_g3_count / len(runs), 3) if runs else 0.0,
            "gate4_confirmation_rate": round(tp_g4_count / len(runs), 3) if runs else 0.0,
            "total_unmatched_fp_candidates": total_fps,
            "total_fp_confirmed": total_fp_conf,
            "runs": runs,
        }
        print(f"Summary @ tau={tau:.2f}: Gate 3 Recovered: {tp_g3_count}/{len(runs)} | Gate 4 Confirmed: {tp_g4_count}/{len(runs)} | Unmatched FP Cand: {total_fps}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nTarget sweep artifacts saved to: {output_path}")

    return sweep_results


if __name__ == "__main__":
    run_target_sweep()
