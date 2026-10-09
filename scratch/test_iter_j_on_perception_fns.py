"""Test Iteration J perception recovery on the 8 FN videos."""

import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from road_surveillance.detection.yolo_detector import YOLO26Detector
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.road_surface import RoadSurfaceDetector
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.video.ingestion import IngestedFrame
from road_surveillance.tracking.track_state import Track, TrackletState
from scripts.run_phase1_final_validation import Phase1ProductionIncidentPipeline
from scripts.benchmark_iteration_b_detector_recovery import evaluate_video_candidates_temporal

FN_VIDEOS = [
    ("82sQ7N-aHOw_00", 15),
    ("Z4kg2Ev3vhk_00", 170),
    ("UarP8qU1S-c_00", 130),
    ("UarP8qU1S-c_01", 30),
    ("VozrseCzhQc_00", 84),
    ("fzWY0vLAXzI_00", 88),
    ("kABWTDKgc9U_00", 197),
    ("nAXTthLfgtI_00", 62),
]

def test_video_iter_j(vid_id: str, gt_frame: int):
    video_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", f"{vid_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return False, None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adaptive input resolution: for >= 1080p, use 1280 to preserve small vehicle features
    imgsz = 1280 if max(w, h) >= 1080 else 640

    print(f"\n==================================================================")
    print(f"Testing Iteration J on {vid_id} (GT={gt_frame}, Res={w}x{h}, imgsz={imgsz}, FPS={fps:.1f})")

    detector = YOLO26Detector("yolo26n.pt", conf_threshold=0.10, input_resolution=imgsz, device="cpu", enable_adaptive_contrast=True)
    road_detector = RoadSurfaceDetector(default_horizon_ratio=0.20)
    tracker = RobustSurveillanceTracker(
        track_activation_threshold=0.20,
        low_confidence_floor=0.10,
        lost_track_buffer=30,
        road_surface_detector=road_detector,
    )
    geom = GeometryEngine(PlanarHomography())
    pipeline = Phase1ProductionIncidentPipeline(is_camera_calibrated=False)

    f = 0
    all_cands = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret: break

        pts_ns = int(f * (1e9 / fps))
        ingested = IngestedFrame(
            camera_id="cam_01",
            video_id=vid_id,
            frame_idx=f,
            pts_ns=pts_ns,
            fps=fps,
            image=frame_bgr,
        )

        batch = detector.detect(ingested)

        # Road surface filtering: remove off-road detections above the horizon line
        f_boxes, f_confs, f_cids, f_labels = road_detector.filter_detections_on_road(
            batch.boxes, batch.confidences, batch.class_ids, batch.class_labels, frame_bgr.shape
        )
        batch.boxes = f_boxes
        batch.confidences = f_confs
        batch.class_ids = f_cids
        batch.class_labels = f_labels

        tracklets, tracks = tracker.update(batch)
        kin, conv = geom.process_frame(tracklets, tracks)

        cands = pipeline.evaluate_frame(f, pts_ns, tracks, kin, conv)
        all_cands.extend(cands)

        f += 1

    cap.release()

    eval_res = evaluate_video_candidates_temporal(
        all_cands, gt_frame, fps, tolerance_frames=35, max_gap_frames=25
    )

    is_tp = eval_res["is_tp"]
    lead_f = eval_res["lead_frames"]
    lead_s = eval_res["lead_sec"]
    fp_ep = eval_res["fp_episodes"]
    raw_cnt = eval_res["raw_candidates_count"]

    lead_str = f"{lead_s:.2f}s" if lead_s is not None else "N/A"
    print(f"  Result: {'>>> TP RECOVERED! <<<' if is_tp else 'STILL FN'}")
    print(f"  Lead: {lead_f} frames ({lead_str}) | FP episodes: {fp_ep} | Raw alarms: {raw_cnt}")
    if all_cands:
        print(f"  Raw triggers ({len(all_cands)}):")
        for c in all_cands:
            diff = c['frame_idx'] - gt_frame
            print(f"    Frame {c['frame_idx']} (rel {diff:+3d}f): {c['detector']} tracks={c['track_ids']}")

    return is_tp, eval_res

if __name__ == "__main__":
    tps = 0
    for vid, gt in FN_VIDEOS:
        is_tp, _ = test_video_iter_j(vid, gt)
        if is_tp:
            tps += 1
    print(f"\n{'='*70}\nIteration J Summary: Recovered {tps}/{len(FN_VIDEOS)} perception FN videos!\n{'='*70}")
