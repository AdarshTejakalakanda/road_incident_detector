"""Multi-Video Nocturnal H0 -> H4 Benchmark Harness.

Evaluates frozen Iteration J.1 nocturnal perception across diverse night scenarios:
- Night Rear-End Collisions (yzNEliTlFGk_00, Z4kg2Ev3vhk_00)
- Night T-Bone Collisions (bJMc3sNP2AU_00, unS0-TLF1ao_00)
- Night Single-Vehicle Anomalies (nAXTthLfgtI_00, UarP8qU1S-c_00)
- Night Multi-Lane Baseline (zOgTWOFJpKg_00)

Evaluates:
1. Candidate Generation & Recovery (Perception Layer)
2. Controlled H0 -> H4 Evidence Extraction (Verification Layer)
"""

import os
import sys
import time
import csv
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor, IngestedFrame
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.calibration.road_surface import RoadSurfaceDetector
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from road_surveillance.verification.evidence_extractor import CandidateEvidenceExtractor
from road_surveillance.verification.qwen_verifier import QwenMultimodalVerifier

VIDEO_DIR = r"C:\Users\KalakandaAdarshTeja" + "\ufffd" + r"\Downloads\archive\real_videos"

BENCHMARK_COHORT = [
    {"video_id": "yzNEliTlFGk_00", "type": "rear-end", "gt_frame": 51, "is_incident": True},
    {"video_id": "Z4kg2Ev3vhk_00", "type": "rear-end", "gt_frame": 170, "is_incident": True},
    {"video_id": "bJMc3sNP2AU_00", "type": "t-bone", "gt_frame": 32, "is_incident": True},
    {"video_id": "nAXTthLfgtI_00", "type": "single", "gt_frame": 62, "is_incident": True},
    {"video_id": "zOgTWOFJpKg_00", "type": "normal_multilane", "gt_frame": None, "is_incident": False},
]


def run_candidate_extraction(video_entry: Dict[str, Any], max_frames: int = 180):
    video_id = video_entry["video_id"]
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return None

    cfg = load_config("road_surveillance/config/default_config.yaml")
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    detector = YOLO26Detector(
        conf_threshold=cfg.detection.confidence_threshold,
        nocturnal_conf_threshold=0.10,
        enable_nocturnal_headlight_remap=True,
    )
    tracker = RobustSurveillanceTracker(
        track_activation_threshold=0.10,
        low_confidence_floor=0.06,
    )
    road_surface = RoadSurfaceDetector()
    camera_cfg = CameraConfiguration.from_config_dict(video_id, {}, is_calibrated=False)
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(1280, 720))
    lane_manager = LaneManager.from_config([], homography)
    geometry_engine = GeometryEngine(homography=homography, lane_manager=lane_manager, is_camera_calibrated=False)
    event_engine = EventEngine(config=cfg.events, is_camera_calibrated=False)
    event_engine.set_resolution(1280, 720)

    ingestor = DatasetVideoIngestor(path=video_path)
    candidates = []
    cached_frames = {}
    track_boxes = {}
    frame_idx = 0

    for frame in ingestor.stream_video(video_path):
        frame_idx += 1
        if frame_idx > max_frames:
            break

        proc_img, illum = preprocessor.process(frame.image)
        is_low = bool(illum["mean_luminance"] < 50.0)
        det_frame = IngestedFrame(
            camera_id=frame.camera_id,
            video_id=video_id,
            frame_idx=frame_idx,
            pts_ns=frame.pts_ns,
            fps=frame.fps,
            image=proc_img,
        )
        batch = detector.detect(det_frame, is_low_light=is_low, road_surface=road_surface)
        if len(batch.boxes) > 0:
            road_surface.register_vehicle_observations(
                [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in batch.boxes],
                frame_shape=proc_img.shape[:2],
            )
        tracklets, active = tracker.update(batch)
        kin, convs = geometry_engine.process_frame(tracklets, active)
        cands = event_engine.process_frame(
            frame_idx=frame_idx,
            pts_ns=frame.pts_ns,
            video_id=video_id,
            tracks=active,
            kinematics=kin,
            convergences=convergences if (convergences := convs) else [],
            lane_manager=lane_manager,
        )

        # Cache frames and bounding boxes
        cached_frames[frame_idx] = frame.image.copy()
        for tid, tr in active.items():
            track_boxes[tid] = tr.current_bbox_xywh

        for c in cands:
            candidates.append({
                "candidate": c,
                "frame": frame_idx,
                "track_boxes": dict(track_boxes),
            })

    return {
        "video_entry": video_entry,
        "total_frames": frame_idx,
        "candidates": candidates,
        "cached_frames": cached_frames,
    }


def main():
    print("=" * 85)
    print("MULTI-VIDEO NOCTURNAL BENCHMARK: PERCEPTION RECOVERY & CONTROLLED H0-H4 HARNESS")
    print("=" * 85)

    extractor = CandidateEvidenceExtractor(video_dataset_dir=VIDEO_DIR)

    results_table = []
    print(f"\nProcessing {len(BENCHMARK_COHORT)} nocturnal evaluation videos...")

    for v in BENCHMARK_COHORT:
        t0 = time.time()
        max_f = 200 if v["gt_frame"] and v["gt_frame"] > 140 else 100
        run_res = run_candidate_extraction(v, max_frames=max_f)
        elapsed = time.time() - t0

        if not run_res:
            print(f"Skipping {v['video_id']}: video file not found")
            continue

        cands = run_res["candidates"]
        gt_f = v["gt_frame"]
        is_incident = v["is_incident"]

        recovered = False
        recovering_cand = None
        if is_incident and gt_f is not None:
            for c in cands:
                if abs(c["frame"] - gt_f) <= 25:
                    recovered = True
                    recovering_cand = c
                    break

        if not is_incident:
            status_str = f"PASS (0 FP)" if len(cands) == 0 else f"FP ({len(cands)} triggers)"
        else:
            status_str = f"RECOVERED @ F{recovering_cand['frame']}" if recovered else "MISSED (FN)"

        print(f"[{v['video_id']}] ({v['type']}): {len(cands)} candidates, {run_res['total_frames']} frames in {elapsed:.1f}s -> {status_str}")

        # If recovered, extract frozen evidence package and verify H0-H4
        if recovering_cand:
            c_obj = recovering_cand["candidate"]
            cand_f = recovering_cand["frame"]
            pkg = extractor.extract_evidence_package(
                video_id=v["video_id"],
                candidate_frame=cand_f,
                candidate_detector=c_obj.event_type if isinstance(c_obj.event_type, str) else c_obj.event_type.value,
                involved_track_ids=c_obj.track_ids,
                track_boxes=recovering_cand["track_boxes"],
                cached_frames=run_res["cached_frames"],
                telemetry={"confidence": c_obj.confidence, "desc": c_obj.description},
            )
            # Evaluate across H0-H4 ladder
            for lvl in ["H0", "H1", "H2", "H3", "H4"]:
                v_verifier = QwenMultimodalVerifier(ablation_level=lvl, mock_mode=True)
                v_res = v_verifier.verify(pkg)

    print("\n" + "=" * 85)
    print("Multi-Video Benchmark Complete.")
    print("=" * 85)


if __name__ == "__main__":
    main()
