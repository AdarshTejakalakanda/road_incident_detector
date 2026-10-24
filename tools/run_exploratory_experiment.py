"""Exploratory Detector-Parameter Optimization Experiment Runner.

Executes a 30-video exploratory sweep across 5 detector arms:
  Control: Production YOLO26
  Candidate A: UVH-26 YOLO11-S
  Candidate B: UVH-26 RT-DETRv2-S
  Candidate C: vietnguyen Highway YOLOv8m
  Candidate D: G2mus Night YOLOv8s (Night cohorts only; Day marked NOT EVALUATED)

Sweeps confidence thresholds tau_conf in {0.10, 0.25} with escalation tau_esc = 0.50.
"""

import os
import sys
import time
import json
import argparse
import psutil
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.detector_ab_adapters import (
    UltralyticsYOLOAdapter,
    RTDETRv2Adapter,
    StandardDetectionOutput,
)
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.config import load_config
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events.engine import EventEngine


ARM_MAP = {
    "Control": ("YOLO26n (Control)", "models/yolo26n.pt", "yolo", {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}),
    "Cand_A": ("UVH-26 YOLO11-S", "models/benchmark_arms/UVH-26-MV-YOLOv11-S.pt", "yolo", {
        0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
        6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
        10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
    }),
    "Cand_B": ("UVH-26 RT-DETRv2-S", "models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth", "rtdetr", {
        0: "car", 1: "car", 2: "car", 3: "van", 4: "bus", 5: "truck",
        6: "auto_rickshaw", 7: "motorcycle", 8: "truck", 9: "bus",
        10: "van", 11: "bicycle", 12: "van", 13: "other_vehicle"
    }),
    "Cand_C": ("vietnguyen Highway YOLOv8m", "models/benchmark_arms/vietnguyen_best.pt", "yolo", {
        0: "auto_rickshaw", 1: "bus", 2: "car", 3: "truck",
        4: "motorcycle", 5: "truck", 6: "other_vehicle", 7: "truck"
    }),
    "Cand_D": ("G2mus Night YOLOv8s", "models/benchmark_arms/night-vehicle-yolov8s.pt", "yolo", {
        0: "car", 1: "truck", 2: "bus"
    }),
}


def load_adapter(arm_key: str):
    name, weights, mtype, unmap = ARM_MAP[arm_key]
    if mtype == "yolo":
        return UltralyticsYOLOAdapter(arm_id=arm_key, model_name=name, weights_path=weights, unified_map=unmap)
    elif mtype == "rtdetr":
        return RTDETRv2Adapter(
            arm_id=arm_key,
            model_name=name,
            config_path="models/benchmark_arms/rtdetr_s_flattened.yaml",
            weights_path=weights,
            unified_map=unmap
        )
    raise ValueError(f"Unknown model type: {mtype}")


def init_pipeline(w: int, h: int, fps: float, conf_th: float, default_calib_cfg: Any):
    """Instantiate isolated tracker, geometry engine, and event engine."""
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id="eval_cam",
        cfg_obj=default_calib_cfg,
        is_calibrated=False,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(w, h))
    lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
    
    tracker = ByteTrackTracker(
        track_activation_threshold=conf_th,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=max(1, int(fps)),
    )
    geometry_engine = GeometryEngine(
        homography=homography,
        lane_manager=lane_manager,
        smoothing_window=5,
        stationary_speed_threshold_kmh=3.0,
        is_camera_calibrated=False,
    )
    event_engine = EventEngine(is_camera_calibrated=False)
    event_engine.set_resolution(w, h)
    
    return tracker, geometry_engine, event_engine, lane_manager


def run_experiment(manifest_path: str, output_path: str, smoke_test: bool = False, eval_window_cap: Optional[int] = 45):
    with open(manifest_path) as f:
        manifest_data = json.load(f)
        
    videos = manifest_data["videos"]
    if smoke_test:
        print("\n*** RUNNING SMOKE TEST ON 1 VIDEO (FIRST 30 FRAMES) ***\n")
        videos = [videos[0]]
        max_frames_override = 30
    else:
        max_frames_override = None

    cfg = load_config()
    default_calib = cfg.calibration.get("default")

    proc = psutil.Process()
    archive_root = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive"
    
    arms_to_run = ["Control", "Cand_A", "Cand_B", "Cand_C", "Cand_D"]
    
    experiment_results = {
        "metadata": {
            "experiment": "30-video exploratory detector-parameter optimization experiment",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "smoke_test": smoke_test,
            "videos_count": len(videos),
            "arms": list(ARM_MAP.keys()),
            "conf_thresholds": [0.10, 0.25],
            "tau_esc": 0.50,
        },
        "per_arm_results": {},
        "system_telemetry": {}
    }

    # Execute one model process at a time
    for arm_key in arms_to_run:
        arm_name = ARM_MAP[arm_key][0]
        print(f"\n=======================================================")
        print(f"Loading Arm [{arm_key}]: {arm_name}")
        print(f"=======================================================")
        
        adapter = load_adapter(arm_key)
        
        # 10 warm-up frames on dummy input
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            _ = adapter.detect(dummy, conf_threshold=0.10, frame_id=-1)
            
        arm_record = {
            "arm_key": arm_key,
            "model_name": arm_name,
            "per_video": []
        }
        
        for v_idx, v_meta in enumerate(videos):
            vid_id = v_meta["clip_id"]
            cohort = v_meta["cohort"]
            video_rel_path = v_meta["path"]
            full_video_path = os.path.join(archive_root, video_rel_path)
            
            # Constraint: Cand_D is Night cohorts only; mark Day as NOT EVALUATED
            if arm_key == "Cand_D" and cohort == "day":
                arm_record["per_video"].append({
                    "clip_id": vid_id,
                    "cohort": cohort,
                    "status": "NOT_EVALUATED",
                    "reason": "Candidate D (G2mus Night-vehicle) restricted to night cohorts only"
                })
                continue
                
            cap = cv2.VideoCapture(full_video_path)
            assert cap.isOpened(), f"Cannot open {full_video_path}"
            
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            acc_frame = v_meta["accident_frame"]
            
            if max_frames_override:
                frames_to_run = min(total_frames, max_frames_override)
            elif eval_window_cap and acc_frame > 0:
                frames_to_run = min(total_frames, acc_frame + eval_window_cap)
            else:
                frames_to_run = total_frames
            
            # Pipelines for both conf thresholds (0.25 primary, 0.10 secondary)
            tr_25, geom_25, evt_25, lane_25 = init_pipeline(w, h, fps, conf_th=0.25, default_calib_cfg=default_calib)
            tr_10, geom_10, evt_10, lane_10 = init_pipeline(w, h, fps, conf_th=0.10, default_calib_cfg=default_calib)
            
            # State tracking
            latencies_decode = []
            latencies_infer = []
            latencies_track_25 = []
            latencies_track_10 = []
            latencies_event_25 = []
            latencies_event_10 = []
            
            cpu_samples = []
            ram_samples_mb = []
            
            dets_25_all = []
            dets_10_all = []
            classes_25 = []
            classes_10 = []
            
            escalated_candidates_25 = []
            escalated_candidates_10 = []
            
            frame_idx = 0
            while cap.isOpened() and frame_idx < frames_to_run:
                t0_dec = time.perf_counter()
                ret, frame = cap.read()
                t1_dec = time.perf_counter()
                if not ret:
                    break
                latencies_decode.append((t1_dec - t0_dec) * 1000.0)
                
                # Single inference pass at 0.10
                t0_inf = time.perf_counter()
                out_10 = adapter.detect(frame, conf_threshold=0.10, frame_id=frame_idx)
                t1_inf = time.perf_counter()
                latencies_infer.append((t1_inf - t0_inf) * 1000.0)
                
                # Derive exact 0.25 subset
                mask_25 = (out_10.scores >= 0.25)
                boxes_25 = out_10.boxes[mask_25]
                scores_25 = out_10.scores[mask_25]
                u_cids_25 = out_10.unified_class_ids[mask_25] if len(out_10.unified_class_ids) > 0 else np.empty((0,), dtype=int)
                u_names_25 = [out_10.unified_class_names[i] for i, m in enumerate(mask_25) if m]
                raw_names_25 = [out_10.class_names[i] for i, m in enumerate(mask_25) if m]
                
                dets_10_all.extend(out_10.scores.tolist())
                dets_25_all.extend(scores_25.tolist())
                classes_10.extend(out_10.class_names)
                classes_25.extend(raw_names_25)
                
                pts_ns = int(frame_idx * (1e9 / fps))
                
                # Pipeline 0.25 (Primary)
                t0_tr25 = time.perf_counter()
                batch_25 = DetectionBatch(
                    video_id=vid_id, frame_idx=frame_idx, pts_ns=pts_ns,
                    boxes=boxes_25, confidences=scores_25, class_ids=u_cids_25, class_labels=u_names_25
                )
                tlets_25, trks_25 = tr_25.update(batch_25)
                t1_tr25 = time.perf_counter()
                latencies_track_25.append((t1_tr25 - t0_tr25) * 1000.0)
                
                t0_ev25 = time.perf_counter()
                kin_25, conv_25 = geom_25.process_frame(tlets_25, trks_25)
                cands_25 = evt_25.process_frame(
                    frame_idx=frame_idx, pts_ns=pts_ns, video_id=vid_id,
                    tracks=trks_25, kinematics=kin_25, convergences=conv_25, lane_manager=lane_25
                )
                for c in cands_25:
                    if c.confidence >= 0.50:  # tau_esc = 0.50
                        escalated_candidates_25.append((frame_idx, c.confidence, c.event_type))
                t1_ev25 = time.perf_counter()
                latencies_event_25.append((t1_ev25 - t0_ev25) * 1000.0)

                # Pipeline 0.10 (Secondary)
                t0_tr10 = time.perf_counter()
                batch_10 = DetectionBatch(
                    video_id=vid_id, frame_idx=frame_idx, pts_ns=pts_ns,
                    boxes=out_10.boxes, confidences=out_10.scores, class_ids=out_10.unified_class_ids, class_labels=out_10.unified_class_names
                )
                tlets_10, trks_10 = tr_10.update(batch_10)
                t1_tr10 = time.perf_counter()
                latencies_track_10.append((t1_tr10 - t0_tr10) * 1000.0)
                
                t0_ev10 = time.perf_counter()
                kin_10, conv_10 = geom_10.process_frame(tlets_10, trks_10)
                cands_10 = evt_10.process_frame(
                    frame_idx=frame_idx, pts_ns=pts_ns, video_id=vid_id,
                    tracks=trks_10, kinematics=kin_10, convergences=conv_10, lane_manager=lane_10
                )
                for c in cands_10:
                    if c.confidence >= 0.50:
                        escalated_candidates_10.append((frame_idx, c.confidence, c.event_type))
                t1_ev10 = time.perf_counter()
                latencies_event_10.append((t1_ev10 - t0_ev10) * 1000.0)

                if frame_idx % 15 == 0:
                    cpu_samples.append(proc.cpu_percent())
                    ram_samples_mb.append(proc.memory_info().rss / (1024 * 1024))
                    
                frame_idx += 1
                
            cap.release()
            
            # Duration & Alert Evaluation (+/- 35 frames around accident_frame)
            actual_duration_s = frame_idx / fps
            pre_impact_duration_s = max(0.0, (acc_frame - 35) / fps)
            
            # Cluster alert triggers (max gap 25 frames)
            def evaluate_alerts(cand_list):
                if not cand_list:
                    return {"tp": 0, "fp": 0, "fn": 1, "alert_count": 0, "fa_per_hour": 0.0}
                triggers = sorted([c[0] for c in cand_list])
                clusters = []
                curr_cl = [triggers[0]]
                for t in triggers[1:]:
                    if t - curr_cl[-1] <= 25:
                        curr_cl.append(t)
                    else:
                        clusters.append(curr_cl[0])
                        curr_cl = [t]
                if curr_cl:
                    clusters.append(curr_cl[0])
                    
                tp = 0
                fp = 0
                for cl_frame in clusters:
                    if abs(cl_frame - acc_frame) <= 35:
                        tp = 1
                    elif cl_frame < acc_frame - 35:
                        fp += 1
                fn = 1 if tp == 0 else 0
                fa_rate = (fp / (pre_impact_duration_s / 3600.0)) if pre_impact_duration_s > 0 else 0.0
                return {"tp": tp, "fp": fp, "fn": fn, "alert_count": len(clusters), "fa_per_hour": round(fa_rate, 2)}
                
            alert_25 = evaluate_alerts(escalated_candidates_25)
            alert_10 = evaluate_alerts(escalated_candidates_10)
            
            vid_res = {
                "clip_id": vid_id,
                "cohort": cohort,
                "frames_processed": frame_idx,
                "duration_s": round(actual_duration_s, 2),
                "fps": round(fps, 1),
                "accident_frame": acc_frame,
                "conf_0_25": {
                    "total_detections": len(dets_25_all),
                    "dets_per_second": round(len(dets_25_all) / actual_duration_s, 2) if actual_duration_s > 0 else 0.0,
                    "max_score": round(max(dets_25_all), 4) if dets_25_all else 0.0,
                    "mean_score": round(float(np.mean(dets_25_all)), 4) if dets_25_all else 0.0,
                    "median_score": round(float(np.median(dets_25_all)), 4) if dets_25_all else 0.0,
                    "class_counts": dict(pd.Series(classes_25).value_counts()) if classes_25 else {},
                    "alerts": alert_25,
                },
                "conf_0_10": {
                    "total_detections": len(dets_10_all),
                    "dets_per_second": round(len(dets_10_all) / actual_duration_s, 2) if actual_duration_s > 0 else 0.0,
                    "max_score": round(max(dets_10_all), 4) if dets_10_all else 0.0,
                    "mean_score": round(float(np.mean(dets_10_all)), 4) if dets_10_all else 0.0,
                    "median_score": round(float(np.median(dets_10_all)), 4) if dets_10_all else 0.0,
                    "class_counts": dict(pd.Series(classes_10).value_counts()) if classes_10 else {},
                    "alerts": alert_10,
                },
                "latency_ms": {
                    "decode_mean": round(float(np.mean(latencies_decode)), 2) if latencies_decode else 0.0,
                    "infer_mean": round(float(np.mean(latencies_infer)), 2) if latencies_infer else 0.0,
                    "infer_p50": round(float(np.percentile(latencies_infer, 50)), 2) if latencies_infer else 0.0,
                    "infer_p95": round(float(np.percentile(latencies_infer, 95)), 2) if latencies_infer else 0.0,
                    "track_mean": round(float(np.mean(latencies_track_25)), 2) if latencies_track_25 else 0.0,
                    "event_mean": round(float(np.mean(latencies_event_25)), 2) if latencies_event_25 else 0.0,
                },
                "resources": {
                    "mean_cpu_pct": round(float(np.mean(cpu_samples)), 1) if cpu_samples else 0.0,
                    "peak_ram_mb": round(float(np.max(ram_samples_mb)), 1) if ram_samples_mb else 0.0,
                }
            }
            arm_record["per_video"].append(vid_res)
            print(f"[{v_idx+1}/{len(videos)}] {vid_id} ({cohort}) -> {frame_idx}f | inf={vid_res['latency_ms']['infer_mean']}ms | dets@25={len(dets_25_all)}, @10={len(dets_10_all)} | alerts@25: TP={alert_25['tp']}, FP={alert_25['fp']}")

        experiment_results["per_arm_results"][arm_key] = arm_record
        # Free adapter memory
        del adapter
        
    def json_default(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=2, default=json_default)
    print(f"\nExperiment complete. Full results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="scratch/exploratory_30_manifest.json")
    parser.add_argument("--output", default="scratch/exploratory_experiment_results.json")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--eval-window-cap", type=int, default=45, help="Cap frames at accident_frame + cap")
    args = parser.parse_args()
    run_experiment(args.manifest, args.output, smoke_test=args.smoke_test, eval_window_cap=args.eval_window_cap)
