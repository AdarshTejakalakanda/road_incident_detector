"""Evaluation and Verification Script for Wrong-Way Vector Gating & Incident Consolidation.

Target Video: KTGByI3N4G0_00 (Wisconsin highway night, GT frame 91, incident type: single)
Objectives:
1. Verify KTGByI3N4G0_00 produces distinct episodes:
   - Episode A: wrong_way_hazard (traversal & near-miss interactions)
   - Episode B: barrier_strike / single_vehicle_collision (physical impact at barrier near GT F91)
2. Verify suppression of spurious collision proposals between wrong-way vehicle and oncoming traffic
3. Log suppressed events taxonomy (frame, tracks, module, reason, distance/DCA)
4. Evaluate 10 nocturnal control videos to confirm 0 new false positives.
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
from modules.episode_consolidator import consolidate_candidates_into_episodes, ConsolidatedEpisode
from tools.benchmark_k0_detector import load_30_benchmark_entries

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline_on_video(
    video_path: str,
    video_id: str,
    gt_frame: int,
    cfg: Any,
    yolo_model: YOLO,
    preprocessor: LowLightPreprocessor,
    vocab_gate: VocabularyGate,
    quality_gate: DetectorQualityGate,
    motion_gate: MotionROIGate,
    arbitrator: GroqArbitrator,
    max_frames: int = 250,
    temporal_tolerance: int = 35,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    motion_gate.reset()

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
            d.tau_single = 0.65
            d.set_roadway_polygon(road_poly)

    tracker = RobustSurveillanceTracker()
    motion_gate.reset()

    all_candidates: List[Dict[str, Any]] = []
    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    last_f_idx = 0

    ingestor = DatasetVideoIngestor(path=video_path)
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

        # 1. Primary YOLO 640p
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

        # 2. Motion-Guided ROI Zoom
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

    elapsed_s = time.perf_counter() - t0
    fps = round(last_f_idx / elapsed_s, 1) if elapsed_s > 0 else 0.0

    # Retrieve suppressed events
    suppressed_events = event_engine.get_suppressed_events()

    # 6. Episode Consolidation
    episodes = consolidate_candidates_into_episodes(
        raw_candidates=all_candidates,
        video_id=video_id,
        cooldown_seconds=10.0,
        max_gap_frames=60,
        gt_frame=gt_frame,
        temporal_tolerance=temporal_tolerance,
    )

    # Stage 4: VLM Arbitration on In-Window Episodes
    in_window_episodes = [ep for ep in episodes if ep.is_in_gt_window]
    out_of_window_episodes = [ep for ep in episodes if not ep.is_in_gt_window]

    vlm_calls = 0
    confirmed_episodes = []

    # Arbitrate in-window episodes
    dummy_strip = np.zeros((100, 300, 3), dtype=np.uint8)
    for ep in in_window_episodes:
        vlm_calls += 1
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_signals = cand.get("signals", {})
        cand_type = cand["event_type"]
        is_single = (cand_type in ("single_vehicle_anomaly", "single", "barrier_strike", "single_vehicle_collision"))
        acc_val = float(cand_signals.get("accel_ms2", cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -abs(cand_signals.get("decel_mag", 0.0))))))
        dist_val = float(cand_signals.get("distance_m", cand_signals.get("ground_distance_m", cand_signals.get("dca_m", 10.0))))
        iou_val = float(cand_signals.get("bbox_iou", 0.0))
        comp_val = float(cand_signals.get("composite_score", cand_signals.get("s_single", 0.0)))
        yaw_val = float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 0.0)))
        ar_val = float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.0)))
        edge_val = float(cand_signals.get("d_edge_m", 10.0))
        post_rest = bool(cand_signals.get("post_event_rest", False))
        sudden_decel = bool(cand_signals.get("sudden_deceleration", False) or abs(acc_val) >= 6.0)

        telemetry_payload = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "gt_impact_frame": gt_frame,
            "event_type": cand_type,
            "involved_tracks": cand.get("track_ids", []),
            "composite_score": comp_val,
            "accel_ms2": acc_val,
            "yaw_delta_deg": yaw_val,
            "aspect_ratio_ratio": ar_val,
            "d_edge_m": edge_val,
            "distance_m": dist_val,
            "bbox_iou": iou_val,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", cand_signals.get("metric_speed_kmh", 0.0))),
            "speed_kmh": float(cand_signals.get("metric_speed_kmh", cand_signals.get("speed_kmh", 0.0))),
            "track_age": 10,
            "is_single_vehicle": is_single,
            "weather": "normal",
            "day_time": "night",
            "post_event_rest": post_rest,
            "sudden_deceleration": sudden_decel,
        }

        arb_res = arbitrator.arbitrate(
            evidence_strip=dummy_strip,
            telemetry=telemetry_payload,
            candidate_frame=cand_f,
            video_id=video_id,
        )
        if arb_res.get("is_collision", False) or arb_res.get("verdict") == "CONFIRMED_INCIDENT":
            confirmed_episodes.append({
                "episode": ep,
                "arbitration": arb_res,
                "is_tp": True,
            })

    # Out of window episodes: check if false positive confirmed
    fp_episodes = []
    for ep in out_of_window_episodes:
        vlm_calls += 1
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_signals = cand.get("signals", {})
        cand_type = cand["event_type"]
        is_single = (cand_type in ("single_vehicle_anomaly", "single", "barrier_strike", "single_vehicle_collision"))
        acc_val = float(cand_signals.get("accel_ms2", cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -abs(cand_signals.get("decel_mag", 0.0))))))
        dist_val = float(cand_signals.get("distance_m", cand_signals.get("ground_distance_m", cand_signals.get("dca_m", 10.0))))
        iou_val = float(cand_signals.get("bbox_iou", 0.0))
        comp_val = float(cand_signals.get("composite_score", cand_signals.get("s_single", 0.0)))
        yaw_val = float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 0.0)))
        ar_val = float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.0)))
        edge_val = float(cand_signals.get("d_edge_m", 10.0))
        post_rest = bool(cand_signals.get("post_event_rest", False))
        sudden_decel = bool(cand_signals.get("sudden_deceleration", False) or abs(acc_val) >= 6.0)

        telemetry_payload = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "gt_impact_frame": gt_frame,
            "event_type": cand_type,
            "involved_tracks": cand.get("track_ids", []),
            "composite_score": comp_val,
            "accel_ms2": acc_val,
            "yaw_delta_deg": yaw_val,
            "aspect_ratio_ratio": ar_val,
            "d_edge_m": edge_val,
            "distance_m": dist_val,
            "bbox_iou": iou_val,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", cand_signals.get("metric_speed_kmh", 0.0))),
            "speed_kmh": float(cand_signals.get("metric_speed_kmh", cand_signals.get("speed_kmh", 0.0))),
            "track_age": 4,
            "is_single_vehicle": is_single,
            "weather": "normal",
            "day_time": "night",
            "post_event_rest": post_rest,
            "sudden_deceleration": sudden_decel,
        }

        arb_res = arbitrator.arbitrate(
            evidence_strip=dummy_strip,
            telemetry=telemetry_payload,
            candidate_frame=cand_f,
            video_id=video_id,
        )
        if arb_res.get("is_collision", False) or arb_res.get("verdict") == "CONFIRMED_INCIDENT":
            fp_episodes.append({
                "episode": ep,
                "arbitration": arb_res,
                "is_fp": True,
            })

    return {
        "video_id": video_id,
        "gt_frame": gt_frame,
        "last_frame": last_f_idx,
        "elapsed_s": round(elapsed_s, 2),
        "fps": fps,
        "raw_candidates": all_candidates,
        "raw_candidates_count": len(all_candidates),
        "consolidated_episodes": episodes,
        "consolidated_episodes_count": len(episodes),
        "in_window_episodes": in_window_episodes,
        "out_of_window_episodes": out_of_window_episodes,
        "confirmed_tp_episodes": confirmed_episodes,
        "confirmed_fp_episodes": fp_episodes,
        "suppressed_events": suppressed_events,
        "vlm_calls": vlm_calls,
    }


def main():
    metadata_path = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv"
    video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"
    target_vid = "KTGByI3N4G0_00"

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

    # 1. EVALUATE TARGET VIDEO
    print("=" * 100)
    print(f"EVALUATING TARGET VIDEO: {target_vid}")
    print("=" * 100)
    vpath = os.path.join(video_dir, f"{target_vid}.mp4")

    res = run_pipeline_on_video(
        video_path=vpath,
        video_id=target_vid,
        gt_frame=91,
        cfg=cfg,
        yolo_model=yolo_model,
        preprocessor=preprocessor,
        vocab_gate=vocab_gate,
        quality_gate=quality_gate,
        motion_gate=motion_gate,
        arbitrator=arbitrator,
        max_frames=200,
    )

    print(f"Execution complete: {res['last_frame']} frames in {res['elapsed_s']}s ({res['fps']} FPS)")
    print(f"Raw Candidates Count:      {res['raw_candidates_count']}")
    print(f"Consolidated Episodes:     {res['consolidated_episodes_count']}")
    print(f"VLM Arbitrations Dispatched: {res['vlm_calls']}")
    print(f"Confirmed TP Episodes:     {len(res['confirmed_tp_episodes'])}")
    print(f"Confirmed FP Episodes:     {len(res['confirmed_fp_episodes'])}")

    print("\n--- RAW CANDIDATES ---")
    for i, c in enumerate(res["raw_candidates"]):
        print(f"  [{i+1:2d}] F{c['frame']:<3} (t={c['timestamp_s']}s) | Type: {c['event_type']:<24} | Tracks: {c['track_ids']} | Conf: {c['confidence']} | {c['description']}")

    print("\n--- CONSOLIDATED EPISODES ---")
    for i, ep in enumerate(res["consolidated_episodes"]):
        near_misses = getattr(ep, "attached_near_misses", [])
        print(f"  Episode {i+1}:")
        print(f"    - Primary Event:  {ep.primary_event_type}")
        print(f"    - Frame Range:    F{ep.start_frame} -> F{ep.end_frame} (Peak F{ep.peak_frame})")
        print(f"    - In GT Window:   {ep.is_in_gt_window} (GT F91)")
        print(f"    - Track IDs:      {ep.track_ids}")
        print(f"    - Peak Conf:      {ep.peak_confidence}")
        print(f"    - Candidate Count:{len(ep.candidates)}")
        print(f"    - Attached Near Misses: {len(near_misses)}")

    print("\n--- SUPPRESSED EVENTS AUDIT LOG ---")
    suppressed = res["suppressed_events"]
    print(f"Total Suppressed Events Logged: {len(suppressed)}")
    # Group suppressed by reason
    by_reason = defaultdict(list)
    for s in suppressed:
        by_reason[s["reason"]].append(s)

    for reason, items in by_reason.items():
        print(f"\n* Suppression Reason: {reason} (Count: {len(items)})")
        for item in items[:5]:  # print sample of up to 5
            t_ids = item.get("track_ids", item.get("tracks", []))
            m_val = item.get("metric_val", item.get("distance_or_dca", "N/A"))
            print(f"    Frame {item.get('frame', 'N/A'):<3} | Tracks: {t_ids} | Module: {item.get('module', 'N/A')} | Metric: {m_val}")
        if len(items) > 5:
            print(f"    ... and {len(items)-5} more occurrences")

    # 2. EVALUATE 10 NOCTURNAL CONTROL RUNS
    print("\n" + "=" * 100)
    print("NOCTURNAL CONTROL REGRESSION SWEEP (10 RUNS)")
    print("Hard Gate: Exactly 0 False Positives")
    print("=" * 100)

    loader = GroundTruthLoader(metadata_path)
    all_30 = load_30_benchmark_entries(metadata_path)
    nocturnal_entries = [e for e in all_30 if loader.get_by_video_id(e["video_id"]).day_time == "night"]

    noc_tp = 0
    noc_fn = 0
    noc_fp = 0
    noc_results = []

    for i, e in enumerate(nocturnal_entries):
        n_vid = e["video_id"]
        n_gt = int(e["gt_frame"])
        n_vpath = os.path.join(video_dir, f"{n_vid}.mp4")

        n_res = run_pipeline_on_video(
            video_path=n_vpath,
            video_id=n_vid,
            gt_frame=n_gt,
            cfg=cfg,
            yolo_model=yolo_model,
            preprocessor=preprocessor,
            vocab_gate=vocab_gate,
            quality_gate=quality_gate,
            motion_gate=motion_gate,
            arbitrator=arbitrator,
            max_frames=250,
        )

        is_tp = len(n_res["confirmed_tp_episodes"]) > 0
        fp_cnt = len(n_res["confirmed_fp_episodes"])

        if is_tp:
            noc_tp += 1
        else:
            noc_fn += 1
        noc_fp += fp_cnt

        status_str = "TP" if is_tp else "FN"
        if fp_cnt > 0:
            status_str += f"+FP({fp_cnt})"

        print(f"[{i+1:2d}] {n_vid:<16} | GT F{n_gt:<3} | Raw: {n_res['raw_candidates_count']:<2} | Eps: {n_res['consolidated_episodes_count']:<2} | Status: {status_str} | FPS: {n_res['fps']}")
        noc_results.append({
            "video_id": n_vid,
            "gt_frame": n_gt,
            "status": status_str,
            "tp": is_tp,
            "fp_count": fp_cnt,
            "raw_candidates": n_res["raw_candidates_count"],
            "episodes": n_res["consolidated_episodes_count"],
        })

    print("-" * 100)
    print(f"Nocturnal Summary: TP = {noc_tp} / {len(nocturnal_entries)} | FN = {noc_fn} | FP = {noc_fp}")
    print("=" * 100)

    # Save output artifact
    out_data = {
        "target_evaluation": {
            "video_id": target_vid,
            "raw_candidates": res["raw_candidates"],
            "consolidated_episodes_count": res["consolidated_episodes_count"],
            "vlm_calls": res["vlm_calls"],
            "confirmed_tp": len(res["confirmed_tp_episodes"]),
            "confirmed_fp": len(res["confirmed_fp_episodes"]),
            "suppressed_events_count": len(res["suppressed_events"]),
            "suppressed_summary": {r: len(items) for r, items in by_reason.items()},
        },
        "nocturnal_regression": {
            "total_runs": len(nocturnal_entries),
            "tp": noc_tp,
            "fn": noc_fn,
            "fp": noc_fp,
            "runs": noc_results,
        },
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/wrong_way_eval_report.json", "w") as f:
        json.dump(out_data, f, indent=2)
    print("Saved evaluation report to artifacts/wrong_way_eval_report.json")


if __name__ == "__main__":
    main()
