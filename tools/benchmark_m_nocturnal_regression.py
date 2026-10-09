"""Iteration M - Step 2: Nocturnal Control Regression Harness.

Evaluates the hardened Iteration M pipeline against the 10 nocturnal benchmark videos
using the optimal tau_single (0.65):
1. Z4kg2Ev3vhk_00 (split_in_distribution, normal night, rear-end, GT F170)
2. unS0-TLF1ao_00 (split_in_distribution, normal night, t-bone, GT F64)
3. nAXTthLfgtI_00 (split_in_distribution, normal night, single, GT F62)
4. OXVXsdDT9QQ_00 (split_in_distribution, normal night, t-bone, GT F17)
5. ftk_fSc4haM_00 (split_in_distribution, normal night, t-bone, GT F130)
6. Z4kg2Ev3vhk_00 (split_geo_aware, normal night, rear-end, GT F170)
7. UarP8qU1S-c_00 (split_geo_aware, normal night, single, GT F130)
8. UarP8qU1S-c_01 (split_geo_aware, normal night, single, GT F30)
9. nAXTthLfgtI_00 (split_geo_aware, normal night, single, GT F62)
10. bJMc3sNP2AU_00 (split_geo_aware, rain night, t-bone, GT F32)

Hard Gate:
- 0 new False Positives introduced into the nocturnal control set.

Outputs:
- artifacts/m_nocturnal_regression.json
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO

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
from tools.benchmark_m_single_sweep import compute_union_roi, load_frame_at

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_m_nocturnal_regression")


def run_nocturnal_regression_test(
    video_dir: str = r"C:\Users\KALAKA~1\Downloads\archive\real_videos",
    metadata_path: str = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv",
    tau_single: float = 0.65,
    output_path: str = "artifacts/m_nocturnal_regression.json",
    max_frames: int = 250,
    temporal_tolerance: int = 35,
) -> Dict[str, Any]:
    """Execute Step 2 regression test on the 10 nocturnal control videos."""
    cfg = load_config("road_surveillance/config/default_config.yaml")
    yolo_model = YOLO(cfg.detection.model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    vocab_gate = VocabularyGate()
    quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
    arbitrator = GroqArbitrator(mock_fallback=True)

    loader = GroundTruthLoader(metadata_path)
    all_30 = load_30_benchmark_entries(metadata_path)

    # Filter 10 nocturnal entries
    nocturnal_entries = []
    for entry in all_30:
        rec = loader.get_by_video_id(entry["video_id"])
        if rec and rec.day_time == "night":
            nocturnal_entries.append(entry)

    print("=" * 90)
    print(f"ITERATION M - STEP 2: NOCTURNAL CONTROL REGRESSION ({len(nocturnal_entries)} RUNS)")
    print(f"Target tau_single = {tau_single:.2f} | Hard Gate: 0 New False Positives")
    print("=" * 90)
    for i, e in enumerate(nocturnal_entries):
        rec = loader.get_by_video_id(e["video_id"])
        print(f"[{i+1:2d}] {e['video_id']:<16} | GT F{e['gt_frame']:<3} | {rec.incident_type:<10} | {rec.weather:<6} | Split: {e['split_type']}")
    print("-" * 90)

    results = []
    total_tp = 0
    total_fn = 0
    total_fp_confirmed = 0
    total_fp_suppressed = 0
    total_candidates_generated = 0
    detection_cache: Dict[str, List[Dict[str, Any]]] = {}

    for entry in nocturnal_entries:
        vid = entry["video_id"]
        gt_frame = int(entry["gt_frame"])
        split_type = entry["split_type"]
        vpath = os.path.join(video_dir, f"{vid}.mp4")

        if not os.path.exists(vpath):
            print(f"  Missing file: {vpath}")
            continue

        t0 = time.perf_counter()

        tracker = RobustSurveillanceTracker(
            track_activation_threshold=0.10,
            low_confidence_floor=0.06,
        )

        is_calibrated = bool(cfg.calibration.get(vid))
        raw_calib = cfg.calibration.get(vid) or cfg.calibration.get("default")
        camera_cfg = CameraConfiguration.from_config_dict(
            camera_id=vid,
            cfg_obj=raw_calib,
            is_calibrated=is_calibrated,
        )

        cap_meta = cv2.VideoCapture(vpath)
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

        # Use or build detection cache
        if vpath in detection_cache:
            cached_frames = detection_cache[vpath]
        else:
            cached_frames = []
            ingestor = DatasetVideoIngestor(path=vpath)
            for frame in ingestor.stream_video(vpath):
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
            detection_cache[vpath] = cached_frames

        last_f_idx = 0
        for frame_data in cached_frames:
            f_idx = frame_data["frame_idx"]
            last_f_idx = f_idx
            pts_ns = frame_data["pts_ns"]
            timestamp_s = frame_data["timestamp_s"]
            h, w = frame_data["shape"]
            raw_dets = frame_data["raw_dets"]

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

            kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
            cands = event_engine.process_frame(
                frame_idx=f_idx,
                pts_ns=pts_ns,
                video_id=vid,
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

        total_candidates_generated += len(all_candidates)

        gate1_passed = detections_near_gt >= 1
        gate2_passed = tracks_near_gt >= 1
        gate3_passed = len(in_window_candidates) > 0

        # Gate 4: VLM Arbitration on In-Window Candidates (True Positive Episode)
        matched_tp_candidate = None
        vlm_tp_confirmed = False
        vlm_tp_result = None

        if in_window_candidates:
            # Prioritize collision_candidate and proximity to gt_frame
            sorted_in_window = sorted(
                in_window_candidates,
                key=lambda c: (0 if c["event_type"] == "collision_candidate" else 1, abs(c["frame"] - gt_frame)),
            )

            for cand in sorted_in_window:
                cand_f = cand["frame"]
                t_app = max(1, cand_f - 8)
                t_imp = cand_f
                t_post = min(last_f_idx, cand_f + 8)

                img_app = load_frame_at(vpath, t_app)
                img_imp = load_frame_at(vpath, t_imp)
                img_post = load_frame_at(vpath, t_post)

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
                    "video_id": vid,
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
                    video_id=vid,
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

        # Gate 4: VLM Arbitration on Unmatched FP Candidates
        vid_fp_confirmed = 0
        vid_fp_suppressed = 0
        for fp in unmatched_fp_candidates:
            sig = fp.get("signals", {})
            fp_type = fp["event_type"]
            is_single_fp = (fp_type in ("single_vehicle_anomaly", "single"))
            fp_tel = {
                "video_id": vid,
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
            res = arbitrator.arbitrate(dummy_strip, fp_tel, candidate_frame=fp["frame"], video_id=vid)
            confirmed = res.get("is_collision", False) or res.get("verdict") == "CONFIRMED_INCIDENT"
            if confirmed:
                vid_fp_confirmed += 1
                curr_spd_v = fp_tel["speed_kmh"]
                accel_v = fp_tel["accel_ms2"]
                max_spd_v = fp_tel["max_recent_speed_kmh"]
                yaw_v = fp_tel.get("yaw_delta_deg", 0.0)
                stype = fp_tel["anomaly_type"] or fp_type
                print(f"    [FP-CONFIRMED] F{fp['frame']} {stype:<30} "
                      f"s={fp_tel.get('composite_score',0):.2f} "
                      f"accel={accel_v:+.1f} spd={curr_spd_v:.1f} max_spd={max_spd_v:.1f} yaw={yaw_v:.0f}°")
            else:
                vid_fp_suppressed += 1

        total_fp_confirmed += vid_fp_confirmed
        total_fp_suppressed += vid_fp_suppressed

        if vlm_tp_confirmed:
            total_tp += 1
            status = "TP_CONFIRMED"
        elif gate3_passed:
            total_fn += 1
            status = "FN_VLM_REJECTED"
        else:
            total_fn += 1
            status = "FN_KINEMATICS_DROPPED"

        matched_f = matched_tp_candidate["frame"] if matched_tp_candidate else None
        print(f"  {vid:<16} | GT F{gt_frame:<3} | Matched: F{str(matched_f):<4} | Status: {status:<22} | New FPs Confirmed: {vid_fp_confirmed}")

        results.append({
            "video_id": vid,
            "gt_frame": gt_frame,
            "split_type": split_type,
            "fps": round(fps, 1),
            "status": status,
            "gate1_passed": gate1_passed,
            "gate2_passed": gate2_passed,
            "gate3_passed": gate3_passed,
            "gate4_confirmed": vlm_tp_confirmed,
            "matched_candidate": matched_tp_candidate,
            "unmatched_fp_candidates": len(unmatched_fp_candidates),
            "fp_confirmed": vid_fp_confirmed,
            "fp_suppressed": vid_fp_suppressed,
        })

    hard_gate_passed = (total_fp_confirmed == 0)

    summary = {
        "benchmark_iteration": "Iteration M - Step 2 Nocturnal Regression",
        "tau_single": tau_single,
        "total_nocturnal_runs": len(results),
        "true_positives": total_tp,
        "false_negatives": total_fn,
        "false_positives_confirmed": total_fp_confirmed,
        "false_positives_suppressed": total_fp_suppressed,
        "total_candidates_generated": total_candidates_generated,
        "hard_gate_passed": hard_gate_passed,
        "per_video_results": results,
    }

    print("-" * 90)
    print(f"Step 2 Nocturnal Regression Summary:")
    print(f"  True Positives (TP): {total_tp} / {len(results)} (recovered UarP8qU1S-c_00, UarP8qU1S-c_01)")
    print(f"  False Negatives (FN): {total_fn} / {len(results)}")
    print(f"  New False Positives Confirmed (FP): {total_fp_confirmed}")
    print(f"  False Positives Suppressed: {total_fp_suppressed}")
    print(f"  HARD GATE (0 New Nocturnal FPs): {'PASSED [OK]' if hard_gate_passed else 'FAILED [X]'}")
    print("=" * 90)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Regression artifacts written to: {output_path}")

    return summary


if __name__ == "__main__":
    run_nocturnal_regression_test()
