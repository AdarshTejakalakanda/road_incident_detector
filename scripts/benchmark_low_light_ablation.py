"""Low-Light Preprocessing Ablation Benchmark (N0 - N3).

Systematically evaluates:
- N0: Baseline (raw untouched video)
- N1: CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- N2: Gamma Correction (gamma = 0.65)
- N3: Adaptive Selection (enhances only when mean Y < 45)

Measures:
- Vehicle recall / detection count
- Unique tracks & track fragmentation
- Collision candidate generation
- False positive episodes (on zOgTWOFJpKg_00 baseline)
- Preprocessing latency and pipeline FPS
"""

import os
import sys
import time
from typing import Dict, List, Any
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor, RollingFrameBuffer
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
from road_surveillance.calibration import LaneManager
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine
from road_surveillance.preprocessing.illumination import LowLightPreprocessor, IlluminationEstimator


TEST_VIDEOS = [
    "yzNEliTlFGk_00.mp4",  # Night highway collision (distant target)
    "zOgTWOFJpKg_00.mp4",  # Night complex multi-lane (FP sensitivity baseline)
    "yzLkvjtWFeI_00.mp4",  # Day intersection T-bone collision
    "acmBtouCwIU_00.mp4",  # Day highway collision
]

VIDEO_DIR = r"C:\Users\KalakandaAdarshTeja" + "\ufffd" + r"\Downloads\archive\real_videos"


def run_single_ablation(video_path: str, mode: str, max_frames: int = 150) -> Dict[str, Any]:
    """Run pipeline on a single video with a specific low-light mode."""
    cfg = load_config("road_surveillance/config/default_config.yaml")
    detector = YOLO26Detector(
        model_path=cfg.detection.model_path,
        device=cfg.detection.device,
        conf_threshold=0.20,
        enable_adaptive_contrast=False,
    )
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    tracker = RobustSurveillanceTracker(track_activation_threshold=0.20)
    
    from road_surveillance.calibration import PlanarHomography, CameraConfiguration
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
    preprocessor = LowLightPreprocessor(mode=mode, low_light_threshold=45.0, gamma=0.65)

    ingestor = DatasetVideoIngestor(path=video_path)

    total_detections = 0
    total_preproc_time_ms = 0.0
    luminances = []
    candidates_count = 0
    unique_tracks_seen = set()
    frames_processed = 0

    t_start = time.perf_counter()

    for frame in ingestor.stream_video(video_path):
        frames_processed += 1
        if frames_processed > max_frames:
            break

        # Low-light preprocessing
        processed_img, telem = preprocessor.process(frame.image)
        total_preproc_time_ms += telem["latency_ms"]
        luminances.append(telem["mean_luminance"])

        # Detection on preprocessed frame
        frame.image = processed_img
        batch = detector.detect(frame)
        total_detections += len(batch.boxes)

        # Tracking
        tracklets, active_tracks = tracker.update(batch)
        for tid in active_tracks:
            unique_tracks_seen.add(tid)

        # Kinematics & Candidates
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
        candidates_count += len(cands)

    total_time_s = time.perf_counter() - t_start
    fps = frames_processed / max_time if (max_time := max(0.001, total_time_s)) else 0.0

    mean_luminance = float(np.mean(luminances)) if luminances else 0.0
    unique_tracks = len(unique_tracks_seen)
    mean_track_length = (total_detections / unique_tracks) if unique_tracks > 0 else 0.0

    return {
        "video": os.path.basename(video_path),
        "mode": mode,
        "frames": frames_processed,
        "mean_luminance": mean_luminance,
        "total_detections": total_detections,
        "unique_tracks": unique_tracks,
        "mean_track_length": mean_track_length,
        "candidates": candidates_count,
        "avg_preproc_ms": total_preproc_time_ms / max(1, frames_processed),
        "fps": fps,
    }


def main():
    modes = ["N0", "N1", "N2", "N3"]
    print("=" * 80)
    print("STEP 2: LOW-LIGHT PREPROCESSING ABLATION BENCHMARK (N0 - N3)")
    print("=" * 80)

    results = []

    for vid_name in TEST_VIDEOS:
        vid_path = os.path.join(VIDEO_DIR, vid_name)
        if not os.path.exists(vid_path):
            print(f"[SKIP] Video not found: {vid_path}")
            continue

        print(f"\nEvaluating: {vid_name} ...")
        for mode in modes:
            res = run_single_ablation(vid_path, mode, max_frames=120)
            results.append(res)
            print(
                f"  [{mode:4s}] Dets: {res['total_detections']:4d} | Tracks: {res['unique_tracks']:2d} | "
                f"Frag: {res['mean_track_length']:4.1f}f/trk | Cands: {res['candidates']:2d} | "
                f"Preproc: {res['avg_preproc_ms']:.2f}ms | FPS: {res['fps']:.1f}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Video':<18} | {'Mode':<5} | {'Mean Y':<6} | {'Dets':<5} | {'Tracks':<6} | {'f/track':<7} | {'Cands':<5} | {'FPS':<5}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['video']:<18} | {r['mode']:<5} | {r['mean_luminance']:<6.1f} | {r['total_detections']:<5d} | "
            f"{r['unique_tracks']:<6d} | {r['mean_track_length']:<7.1f} | {r['candidates']:<5d} | {r['fps']:<5.1f}"
        )


if __name__ == "__main__":
    main()
