"""Forensic extraction script for 19 Path-C triggers on 5uh8-3q51uI_00.
Strictly READ-ONLY: extracts track histories and computes multi-frame heading metrics.
"""

import os
import sys
import math
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate

PATH_C_RECORDS = [
    {"rec_id": 1, "frame": 21, "tracks": [4, 5]},
    {"rec_id": 2, "frame": 169, "tracks": [45, 50]},
    {"rec_id": 3, "frame": 197, "tracks": [49, 77]},
    {"rec_id": 4, "frame": 291, "tracks": [71, 97]},
    {"rec_id": 5, "frame": 376, "tracks": [100, 109]},
    {"rec_id": 6, "frame": 418, "tracks": [136, 141]},
    {"rec_id": 7, "frame": 426, "tracks": [104, 130]},
    {"rec_id": 8, "frame": 447, "tracks": [110, 117]},
    {"rec_id": 9, "frame": 512, "tracks": [163, 169]},
    {"rec_id": 10, "frame": 602, "tracks": [192, 196]},
    {"rec_id": 11, "frame": 626, "tracks": [196, 201]},
    {"rec_id": 12, "frame": 713, "tracks": [132, 208]},
    {"rec_id": 13, "frame": 716, "tracks": [132, 197]},
    {"rec_id": 14, "frame": 722, "tracks": [217, 232]},
    {"rec_id": 15, "frame": 728, "tracks": [157, 213]},
    {"rec_id": 16, "frame": 779, "tracks": [247, 250]},
    {"rec_id": 17, "frame": 798, "tracks": [247, 257]},
    {"rec_id": 18, "frame": 835, "tracks": [229, 246]},
    {"rec_id": 19, "frame": 880, "tracks": [253, 259]},
]

TARGET_FRAMES = set(x["frame"] for x in PATH_C_RECORDS)
# We want history +/- 30 frames around each trigger
MAX_FRAME = min(899, max(TARGET_FRAMES) + 31)

def main():
    video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\5uh8-3q51uI_00.mp4"
    config_path = "road_surveillance/config/default_config.yaml"
    cfg = load_config(config_path)

    cap_meta = cv2.VideoCapture(video_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()

    is_calibrated = bool(cfg.calibration.get("5uh8-3q51uI_00"))
    raw_calib = cfg.calibration.get("5uh8-3q51uI_00") or cfg.calibration.get("default")
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id="5uh8-3q51uI_00",
        cfg_obj=raw_calib,
        is_calibrated=is_calibrated,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
    lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
    geometry_engine = GeometryEngine(homography=homography, lane_manager=lane_manager, smoothing_window=5)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    model_path = getattr(cfg.detection, "model_path", "yolov8x.pt")
    yolo_model = YOLO(model_path)
    vocab_gate = VocabularyGate()
    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
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
    tracker = RobustSurveillanceTracker()

    # Frame cache: frame_idx -> {tid: {"bbox": bbox, "heading": heading, "speed_mps": speed_mps, "ground_pos": ground_pos}}
    frame_track_data: Dict[int, Dict[int, Any]] = {}
    frame_pts: Dict[int, float] = {}

    ingestor = DatasetVideoIngestor(path=video_path)
    print(f"Tracking video up to frame {MAX_FRAME}...")

    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        if f_idx > MAX_FRAME:
            break

        img = frame.image
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)
        frame_pts[f_idx] = timestamp_s

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

        recovered_micro, _ = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )
        raw_dets.extend(recovered_micro)

        admitted, _ = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )

        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)

        current_frame_dict = {}
        for tid, tr in active_tracks.items():
            k = kinematics.get(tid)
            if k is not None:
                current_frame_dict[tid] = {
                    "bbox": tr.current_bbox_xywh,
                    "heading_deg": float(k.heading_deg),
                    "speed_mps": float(k.speed_mps),
                    "speed_kmh": float(k.speed_kmh),
                    "ground_pos": tuple(k.ground_pos),
                    "class_label": tr.class_label,
                }
        frame_track_data[f_idx] = current_frame_dict

        if f_idx % 100 == 0:
            print(f"Processed frame {f_idx}/{MAX_FRAME}...")

    print("Tracking complete! Now evaluating 19 Path-C records...")

    results = []
    def angular_diff(a1, a2):
        d = abs(a1 - a2) % 360.0
        return min(d, 360.0 - d)

    for rec in PATH_C_RECORDS:
        r_id = rec["rec_id"]
        t = rec["frame"]
        t1, t2 = rec["tracks"]

        # Track data at trigger frame
        d_curr_1 = frame_track_data.get(t, {}).get(t1)
        d_curr_2 = frame_track_data.get(t, {}).get(t2)

        # Planar distance
        if d_curr_1 and d_curr_2:
            p1 = d_curr_1["ground_pos"]
            p2 = d_curr_2["ground_pos"]
            planar_dist = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
        else:
            planar_dist = 0.0

        # We evaluate both tracks to see which one (or both) triggered Path C
        per_track_metrics = {}
        for tid in [t1, t2]:
            # Extract heading series across [t - 30, t + 30]
            series_f = []
            series_h = []
            series_v = []
            for f in range(max(0, t - 30), min(MAX_FRAME, t + 31)):
                if f in frame_track_data and tid in frame_track_data[f]:
                    series_f.append(f)
                    series_h.append(frame_track_data[f][tid]["heading_deg"])
                    series_v.append(frame_track_data[f][tid]["speed_mps"])

            if t not in series_f:
                continue

            idx_t = series_f.index(t)

            # Instantaneous heading change: |theta_t - theta_{t-1}|
            if idx_t >= 1 and series_f[idx_t - 1] == t - 1:
                delta_theta_inst = angular_diff(series_h[idx_t], series_h[idx_t - 1])
                dt_inst = frame_pts[t] - frame_pts[t - 1]
                yaw_rate_inst = delta_theta_inst / max(1e-4, dt_inst)
            else:
                delta_theta_inst = 0.0
                dt_inst = 1.0 / 30.0
                yaw_rate_inst = 0.0

            # 3-frame net angular displacement |theta_t - theta_{t-2}|
            if idx_t >= 2 and series_f[idx_t - 2] == t - 2:
                delta_theta_3f = angular_diff(series_h[idx_t], series_h[idx_t - 2])
            else:
                delta_theta_3f = 0.0

            # 5-frame net angular displacement |theta_t - theta_{t-4}|
            if idx_t >= 4 and series_f[idx_t - 4] == t - 4:
                delta_theta_5f = angular_diff(series_h[idx_t], series_h[idx_t - 4])
            else:
                delta_theta_5f = 0.0

            # 10-frame net angular displacement |theta_t - theta_{t-9}|
            if idx_t >= 9 and series_f[idx_t - 9] == t - 9:
                delta_theta_10f = angular_diff(series_h[idx_t], series_h[idx_t - 9])
            else:
                delta_theta_10f = 0.0

            # N_persist: count of consecutive frames ending at t (or centered at t) where |dtheta/dt| >= 15.0 deg/s
            # Compute frame-to-frame yaw rates for continuous sequence
            rates = []
            for k in range(1, len(series_f)):
                df = series_f[k] - series_f[k - 1]
                if df == 1:
                    dh = angular_diff(series_h[k], series_h[k - 1])
                    dt_k = frame_pts[series_f[k]] - frame_pts[series_f[k - 1]]
                    rates.append(dh / max(1e-4, dt_k))
                else:
                    rates.append(0.0)

            # Find consecutive run of rates >= 15.0 ending at idx_t - 1 (which corresponds to transition t-1 -> t)
            # rate index k-1 corresponds to transition k-1 -> k
            target_rate_idx = idx_t - 1
            n_persist = 0
            if 0 <= target_rate_idx < len(rates):
                # Count backwards from target_rate_idx
                p = target_rate_idx
                while p >= 0 and rates[p] >= 15.0:
                    n_persist += 1
                    p -= 1

            # Lateral acceleration: a_lat = v * yaw_rate (in rad/s)
            v_curr = series_v[idx_t]
            yaw_rate_rad = math.radians(yaw_rate_inst)
            a_lat = v_curr * yaw_rate_rad

            per_track_metrics[tid] = {
                "delta_theta_inst": delta_theta_inst,
                "yaw_rate_inst": yaw_rate_inst,
                "delta_theta_3f": delta_theta_3f,
                "delta_theta_5f": delta_theta_5f,
                "delta_theta_10f": delta_theta_10f,
                "n_persist": n_persist,
                "v_mps": v_curr,
                "a_lat": a_lat,
            }

        # Select the track that had the higher yaw rate
        t_trigger = None
        if t1 in per_track_metrics and t2 in per_track_metrics:
            if per_track_metrics[t1]["yaw_rate_inst"] >= per_track_metrics[t2]["yaw_rate_inst"]:
                t_trigger = t1
            else:
                t_trigger = t2
        elif t1 in per_track_metrics:
            t_trigger = t1
        elif t2 in per_track_metrics:
            t_trigger = t2

        m = per_track_metrics.get(t_trigger, {})
        results.append({
            "rec_id": r_id,
            "frame": t,
            "tracks": [t1, t2],
            "trigger_track": t_trigger,
            "planar_dist": planar_dist,
            "delta_theta_inst": m.get("delta_theta_inst", 0.0),
            "yaw_rate_inst": m.get("yaw_rate_inst", 0.0),
            "delta_theta_3f": m.get("delta_theta_3f", 0.0),
            "delta_theta_5f": m.get("delta_theta_5f", 0.0),
            "delta_theta_10f": m.get("delta_theta_10f", 0.0),
            "n_persist": m.get("n_persist", 0),
            "v_mps": m.get("v_mps", 0.0),
            "a_lat": m.get("a_lat", 0.0),
        })

    with open("scratch/path_c_forensic_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Forensic analysis complete! Results written to scratch/path_c_forensic_results.json")

if __name__ == "__main__":
    main()
