"""Test perception recovery on the 8 FN videos using updated YOLO26 + ByteTrack."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("."))

from road_surveillance.config import load_config, CollisionConfig
from road_surveillance.detection.yolo_detector import YOLO26Detector
from road_surveillance.detection.detector_interface import DetectionBatch
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.tracking.track_state import Track
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.geometry.engine import GeometryEngine
from scripts.run_phase1_final_validation import Phase1ProductionIncidentPipeline
from scripts.benchmark_iteration_b_detector_recovery import (
    compute_iou,
    xyxy_to_xywh,
    evaluate_video_candidates_temporal,
    cluster_episodes,
)

# Videos to test
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

class EnhancedSelectiveRecoveryDetector:
    """Selective detector with mature vehicle recovery + visual conflict recovery."""

    def __init__(self, detector: YOLO26Detector, normal_threshold: float = 0.25, recovery_floor: float = 0.15):
        self.detector = detector
        self.normal_threshold = normal_threshold
        self.recovery_floor = recovery_floor

    def detect_selective(self, frame, current_tracks: Dict[int, Track]) -> DetectionBatch:
        prev_conf = self.detector.conf_threshold
        self.detector.conf_threshold = self.recovery_floor
        batch = self.detector.detect(frame)
        self.detector.conf_threshold = prev_conf

        if len(batch.boxes) == 0:
            return batch

        active_mature_tracks = [t for t in current_tracks.values() if t.status in ("active", "coasting") and t.track_age >= 2]
        eligible_tracks: Set[int] = set()

        # Pairwise conflict tracks
        for i, t1 in enumerate(active_mature_tracks):
            b1 = t1.current_bbox_xywh
            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            L1 = max(15.0, max(b1[2], b1[3]))

            for t2 in active_mature_tracks[i + 1:]:
                b2 = t2.current_bbox_xywh
                c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
                L2 = max(15.0, max(b2[2], b2[3]))

                iou = compute_iou(b1, b2)
                d_px = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                norm_d = d_px / ((L1 + L2) / 2.0)

                if iou >= 0.05 or norm_d <= 1.5:
                    eligible_tracks.add(t1.track_id)
                    eligible_tracks.add(t2.track_id)

        # Also include mature vehicle tracks for single-vehicle recovery (barrier impact / rollover / deformation)
        for t in active_mature_tracks:
            if t.track_age >= 3 and t.get_effective_label() in Track.VEHICLE_CLASSES:
                eligible_tracks.add(t.track_id)

        keep_indices = []
        for idx in range(len(batch.boxes)):
            conf = float(batch.confidences[idx])
            box_xywh = xyxy_to_xywh(batch.boxes[idx])

            if conf >= self.normal_threshold:
                keep_indices.append(idx)
            else:
                recovered = False
                for tid in eligible_tracks:
                    tr = current_tracks.get(tid)
                    if not tr:
                        continue
                    iou = compute_iou(box_xywh, tr.current_bbox_xywh)
                    c_det = (box_xywh[0] + box_xywh[2] / 2.0, box_xywh[1] + box_xywh[3] / 2.0)
                    c_tr = (tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2] / 2.0, tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3] / 2.0)
                    dist_px = np.hypot(c_det[0] - c_tr[0], c_det[1] - c_tr[1])
                    char_len = max(tr.current_bbox_xywh[2], tr.current_bbox_xywh[3])

                    if iou >= 0.20 or dist_px <= (0.8 * char_len):
                        recovered = True
                        break

                if recovered:
                    keep_indices.append(idx)

        if len(keep_indices) == len(batch.boxes):
            return batch

        if not keep_indices:
            return DetectionBatch(
                video_id=batch.video_id,
                frame_idx=batch.frame_idx,
                pts_ns=batch.pts_ns,
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        indices = np.array(keep_indices, dtype=np.int32)
        return DetectionBatch(
            video_id=batch.video_id,
            frame_idx=batch.frame_idx,
            pts_ns=batch.pts_ns,
            boxes=batch.boxes[indices],
            confidences=batch.confidences[indices],
            class_ids=batch.class_ids[indices],
            class_labels=[batch.class_labels[i] for i in keep_indices],
        )


def test_video(vid_id: str, gt_frame: int):
    video_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", f"{vid_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist!")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"\n=======================================================")
    print(f"Processing {vid_id}: GT={gt_frame}, Total={total_frames}, FPS={fps:.1f}")

    from road_surveillance.video.ingestion import IngestedFrame

    base_yolo = YOLO26Detector("yolo26n.pt", conf_threshold=0.25, device="cpu", enable_adaptive_contrast=True)
    selective_det = EnhancedSelectiveRecoveryDetector(base_yolo, normal_threshold=0.25, recovery_floor=0.15)
    tracker = ByteTrackTracker(track_activation_threshold=0.20, frame_rate=int(fps))
    h = PlanarHomography()
    geom = GeometryEngine(h)
    pipeline = Phase1ProductionIncidentPipeline(is_camera_calibrated=False)

    frame_idx = 0
    raw_candidates = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pts_ns = int(frame_idx * (1e9 / fps))
        ingested = IngestedFrame(
            camera_id="cam_01",
            video_id=vid_id,
            frame_idx=frame_idx,
            pts_ns=pts_ns,
            fps=fps,
            image=frame_bgr,
        )

        batch = selective_det.detect_selective(ingested, tracker.tracks)
        tl, tracks = tracker.update(batch)
        kin, conv = geom.process_frame(tl, tracks)

        cands = pipeline.evaluate_frame(frame_idx, pts_ns, tracks, kin, conv)
        raw_candidates.extend(cands)

        frame_idx += 1

    cap.release()

    eval_res = evaluate_video_candidates_temporal(
        raw_candidates, gt_frame, fps, tolerance_frames=35, max_gap_frames=25
    )

    is_tp = eval_res["is_tp"]
    lead_f = eval_res["lead_frames"]
    lead_s = eval_res["lead_sec"]
    fp_ep = eval_res["fp_episodes"]
    raw_cnt = eval_res["raw_candidates_count"]

    lead_str = f"{lead_s:.2f}s" if lead_s is not None else "N/A"
    print(f"  Lead: {lead_f} frames ({lead_str}) | FP episodes: {fp_ep} | Raw alarms: {raw_cnt}")
    if raw_candidates:
        print(f"  Raw triggers:")
        for c in raw_candidates:
            diff = c['frame_idx'] - gt_frame
            print(f"    Frame {c['frame_idx']} (rel {diff:+3d}f): {c['detector']} tracks={c['track_ids']}")
    else:
        print("  No triggers produced.")

    return is_tp, eval_res

if __name__ == "__main__":
    recovered_count = 0
    for vid, gt in FN_VIDEOS:
        is_tp, _ = test_video(vid, gt)
        if is_tp:
            recovered_count += 1

    print(f"\n=======================================================")
    print(f"Summary: Recovered {recovered_count}/{len(FN_VIDEOS)} perception FN videos!")
