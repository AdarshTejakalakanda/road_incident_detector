"""Test SingleVehicleAnomalyDetector on UarP8qU1S-c_00 with RobustSurveillanceTracker."""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from ultralytics import YOLO
from scratch.test_custom_tracker import RobustSurveillanceTracker
from road_surveillance.tracking.track_state import Track, TrackletState
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.events.candidates.collision import CollisionCandidateDetector
from road_surveillance.events.candidates.distant_conflict import DistantTrajectoryConflictDetector
from scripts.run_phase1_final_validation import Phase1ProductionIncidentPipeline
from scripts.benchmark_iteration_b_detector_recovery import evaluate_video_candidates_temporal

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\UarP8qU1S-c_00.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

tracker = RobustSurveillanceTracker(high_threshold=0.20, low_threshold=0.10)
h = PlanarHomography()
geom = GeometryEngine(h)
pipeline = Phase1ProductionIncidentPipeline(is_camera_calibrated=False)

f = 0
all_cands = []

while True:
    ret, frame = cap.read()
    if not ret: break

    pts_ns = int(f * (1e9 / fps))
    p = model.predict(frame, conf=0.10, verbose=False)[0]
    b = p.boxes.xyxy.cpu().numpy()
    c = p.boxes.conf.cpu().numpy()
    cl = p.boxes.cls.cpu().numpy().astype(int)
    names = [model.names[x] for x in cl]

    tracks = tracker.update(b, c, cl, names, pts_ns, frame.shape)

    # Convert to tracklet list for geometry engine
    tracklets = []
    for tid, tr in tracks.items():
        if tr.status in ("active", "coasting"):
            tl = TrackletState(
                video_id="UarP8qU1S-c_00",
                track_id=tid,
                class_id=tr.class_id,
                class_label=tr.get_effective_label(),
                frame_idx=f,
                pts_ns=pts_ns,
                bbox_xywh=tr.current_bbox_xywh,
                ground_contact_px=Track.calculate_ground_contact(tr.current_bbox_xywh),
                track_status=tr.status,
            )
            tracklets.append(tl)

    kin, conv = geom.process_frame(tracklets, tracks)
    cands = pipeline.evaluate_frame(f, pts_ns, tracks, kin, conv)
    all_cands.extend(cands)

    if cands:
        print(f"Trigger at Frame {f}: {cands}")

    f += 1

cap.release()

eval_res = evaluate_video_candidates_temporal(
    all_cands, 130, fps, tolerance_frames=35, max_gap_frames=25
)
print("=" * 60)
print(f"UarP8qU1S-c_00 EVAL RESULT: is_tp={eval_res['is_tp']}, lead_f={eval_res['lead_frames']}, fp_ep={eval_res['fp_episodes']}")
