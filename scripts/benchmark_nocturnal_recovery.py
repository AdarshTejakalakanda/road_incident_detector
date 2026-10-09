"""Benchmark Nocturnal Perception Recovery (Iteration J.1).

Evaluates detector threshold ladder:
- 0.25 (Production Default)
- 0.15
- 0.10
- 0.06

Instruments 4-stage pipeline telemetry:
1. YOLO raw vehicle detections
2. Tracker tracks
3. Mature tracks (age >= 3, real obs >= 2)
4. Incident candidates

Tested on:
- yzNEliTlFGk_00.mp4 (Night rear-end collision, GT frame ~51)
- zOgTWOFJpKg_00.mp4 (Night multi-lane FP baseline)
"""

import os
import sys
import time
from typing import Dict, List, Any, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor, IngestedFrame
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
from road_surveillance.tracking.track_state import Track
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.calibration.road_surface import RoadSurfaceDetector
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine
from road_surveillance.preprocessing.illumination import LowLightPreprocessor


VIDEO_DIR = r"C:\Users\KalakandaAdarshTeja" + "\ufffd" + r"\Downloads\archive\real_videos"
TEST_VIDEOS = {
    "collision": os.path.join(VIDEO_DIR, "yzNEliTlFGk_00.mp4"),
    "fp_baseline": os.path.join(VIDEO_DIR, "zOgTWOFJpKg_00.mp4"),
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}
HEADLIGHT_CLASSES = {"traffic light", "airplane"}


def run_threshold_eval(
    video_path: str,
    vehicle_conf_threshold: float,
    default_conf_threshold: float = 0.25,
    enable_road_gating: bool = True,
    enable_headlight_remap: bool = True,
    max_frames: int = 250,
) -> Dict[str, Any]:
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cfg = load_config("road_surveillance/config/default_config.yaml")

    # Ingestor
    ingestor = DatasetVideoIngestor(path=video_path)

    # Preprocessor (N3 Adaptive)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)

    # Base detector runs at min threshold, class filtering applied selectively
    base_conf = min(vehicle_conf_threshold, default_conf_threshold)
    detector = YOLO26Detector(
        model_path=cfg.detection.model_path,
        device=cfg.detection.device,
        conf_threshold=base_conf,
        enable_adaptive_contrast=False,
    )

    # Tracker: activation threshold tracks vehicle threshold
    tracker_activation = max(0.08, min(vehicle_conf_threshold, 0.20))
    tracker = RobustSurveillanceTracker(
        track_activation_threshold=tracker_activation,
        low_confidence_floor=max(0.04, vehicle_conf_threshold * 0.6),
    )

    # Calibration & Engines
    is_calibrated = bool(cfg.calibration.get(video_id))
    raw_calib_cfg = cfg.calibration.get(video_id) or cfg.calibration.get("default")
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id=video_id,
        cfg_obj=raw_calib_cfg,
        is_calibrated=is_calibrated,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(1280, 720))
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
    event_engine.set_resolution(1280, 720)

    road_surface = RoadSurfaceDetector()

    # Telemetry tracking
    frame_telemetry = []
    total_raw_veh_dets = 0
    unique_tracks = set()
    all_candidates = []
    frames_processed = 0

    for frame in ingestor.stream_video(video_path):
        frames_processed += 1
        if frames_processed > max_frames:
            break

        # 1. Illumination Preprocessing
        proc_img, illum = preprocessor.process(frame.image)
        is_low_light = bool(illum["mean_luminance"] < 50.0)

        # 2. YOLO Raw Detection
        detection_frame = IngestedFrame(
            camera_id=frame.camera_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            pts_ns=frame.pts_ns,
            fps=frame.fps,
            image=proc_img,
        )
        raw_batch = detector.detect(detection_frame)

        # 3. Class-Selective & Roadway Filtering
        filtered_boxes = []
        filtered_confs = []
        filtered_cids = []
        filtered_labels = []

        h_img, w_img = proc_img.shape[:2]
        horizon_y = road_surface.get_horizon_y(h_img)

        raw_veh_count = 0
        for b_box, b_conf, b_cid, b_lbl in zip(
            raw_batch.boxes, raw_batch.confidences, raw_batch.class_ids, raw_batch.class_labels
        ):
            lbl_lower = b_lbl.lower()
            conf_val = float(b_conf)
            bx1, by1, bx2, by2 = b_box
            gc_y = by2

            # Off-road / sky rejection
            if enable_road_gating and gc_y < horizon_y:
                continue

            # Determine threshold by class & low-light condition
            if is_low_light and (lbl_lower in VEHICLE_CLASSES):
                req_conf = vehicle_conf_threshold
            elif is_low_light and enable_headlight_remap and (lbl_lower in HEADLIGHT_CLASSES):
                # Headlight remapping if on road surface
                req_conf = vehicle_conf_threshold
                lbl_lower = "car"
            else:
                req_conf = default_conf_threshold

            if conf_val >= req_conf:
                filtered_boxes.append(b_box)
                filtered_confs.append(conf_val)
                filtered_cids.append(b_cid)
                filtered_labels.append(lbl_lower)
                if lbl_lower in VEHICLE_CLASSES:
                    raw_veh_count += 1

        total_raw_veh_dets += raw_veh_count

        # Build filtered batch
        batch = IngestedFrame(
            camera_id=frame.camera_id,
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            pts_ns=frame.pts_ns,
            fps=frame.fps,
            image=proc_img,
        )
        raw_batch.boxes = np.array(filtered_boxes, dtype=np.float32) if filtered_boxes else np.empty((0, 4), dtype=np.float32)
        raw_batch.confidences = np.array(filtered_confs, dtype=np.float32) if filtered_confs else np.empty((0,), dtype=np.float32)
        raw_batch.class_ids = np.array(filtered_cids, dtype=np.int32) if filtered_cids else np.empty((0,), dtype=np.int32)
        raw_batch.class_labels = filtered_labels

        # Update road surface accumulator
        if len(filtered_boxes) > 0:
            road_surface.register_vehicle_observations(
                [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in filtered_boxes],
                frame_shape=(h_img, w_img),
            )

        # 4. Tracker Update
        tracklets, active_tracks = tracker.update(raw_batch)
        for tid in active_tracks:
            unique_tracks.add(tid)

        # 5. Mature Tracks Check
        mature_count = sum(
            1 for tr in active_tracks.values()
            if tr.track_age >= 3 and sum(1 for p in tr.provenance_history if p == "REAL") >= 2
        )

        # 6. Kinematics & Candidate Events
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
        cands = event_engine.process_frame(
            frame_idx=frame.frame_idx,
            pts_ns=frame.pts_ns,
            video_id=video_id,
            tracks=active_tracks,
            kinematics=kinematics,
            convergences=convergences,
            lane_manager=lane_manager,
        )

        for c in cands:
            etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
            all_candidates.append({
                "frame": frame.frame_idx,
                "type": etype,
                "tracks": c.track_ids,
                "confidence": round(float(c.confidence), 2),
                "description": c.description,
            })

        # Frame telemetry sample around collision (frames 45-65)
        if 45 <= frame.frame_idx <= 65:
            frame_telemetry.append({
                "frame": frame.frame_idx,
                "raw_veh": raw_veh_count,
                "tracker_active": len(active_tracks),
                "mature_tracks": mature_count,
                "candidates": len(cands),
            })

    return {
        "threshold": vehicle_conf_threshold,
        "frames_processed": frames_processed,
        "total_raw_veh_dets": total_raw_veh_dets,
        "unique_tracks": len(unique_tracks),
        "total_candidates": len(all_candidates),
        "candidates": all_candidates,
        "sample_telemetry": frame_telemetry,
    }


def main():
    print("=" * 80)
    print("ITERATION J.1: NOCTURNAL PERCEPTION RECOVERY - THRESHOLD LADDER BENCHMARK")
    print("=" * 80)

    thresholds = [0.25, 0.15, 0.10, 0.06]

    print("\n>>> PART 1: Collision Video (yzNEliTlFGk_00.mp4 - GT Rear-End Crash ~Frame 51)")
    print(f"{'Thresh':<8} | {'Raw Veh':<10} | {'Tracks':<8} | {'Mature (Avg)':<14} | {'Candidates':<12} | {'Collision Recovered?'}")
    print("-" * 80)

    collision_results = {}
    for th in thresholds:
        res = run_threshold_eval(
            video_path=TEST_VIDEOS["collision"],
            vehicle_conf_threshold=th,
            max_frames=120,
        )
        collision_results[th] = res

        # Check if candidate near frame 51 was recovered
        cands_around_crash = [c for c in res["candidates"] if 45 <= c["frame"] <= 75]
        recovered = "YES" if len(cands_around_crash) > 0 else "NO"
        if recovered == "YES":
            rec_types = "/".join(set(c["type"] for c in cands_around_crash))
            recovered += f" ({rec_types} @ F{cands_around_crash[0]['frame']})"

        avg_mature = np.mean([t["mature_tracks"] for t in res["sample_telemetry"]]) if res["sample_telemetry"] else 0.0

        print(f"{th:<8.2f} | {res['total_raw_veh_dets']:<10} | {res['unique_tracks']:<8} | {avg_mature:<14.1f} | {res['total_candidates']:<12} | {recovered}")

    print("\n>>> Detailed Frame Telemetry Around Crash (Frames 48-58) for Collision Video:")
    for th in [0.25, 0.10, 0.06]:
        print(f"\n--- Threshold {th:.2f} ---")
        for t in collision_results[th]["sample_telemetry"]:
            if 48 <= t["frame"] <= 58:
                print(f"  Frame {t['frame']:02d}: YOLO raw veh={t['raw_veh']}, Tracker active={t['tracker_active']}, Mature={t['mature_tracks']}, Candidates={t['candidates']}")

    print("\n" + "=" * 80)
    print(">>> PART 2: FP Baseline Video (zOgTWOFJpKg_00.mp4 - Night Multi-Lane Baseline)")
    print(f"{'Thresh':<8} | {'Raw Veh':<10} | {'Tracks':<8} | {'Total Candidates':<18} | {'Candidate Details'}")
    print("-" * 80)

    for th in thresholds:
        res = run_threshold_eval(
            video_path=TEST_VIDEOS["fp_baseline"],
            vehicle_conf_threshold=th,
            max_frames=120,
        )
        cand_summary = f"{res['total_candidates']} triggers"
        if res["total_candidates"] > 0:
            types = set(c["type"] for c in res["candidates"])
            cand_summary += f" ({', '.join(types)})"
        print(f"{th:<8.2f} | {res['total_raw_veh_dets']:<10} | {res['unique_tracks']:<8} | {res['total_candidates']:<18} | {cand_summary}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
