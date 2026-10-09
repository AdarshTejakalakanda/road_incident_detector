"""Diagnostic script: Offline 5-frame kinematic replay for the 11 surviving distant conflicts.

Strict diagnostic mode: Does NOT modify any production files or configs.
Replays track trajectories from the exact pipeline tracking state on clip 5uh8-3q51uI_00.
"""

import os
import sys
import math
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

# Add workspace to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.geometry.spatial_metrics import compute_trajectory_convergence
from road_surveillance.events.engine import EventEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate

SURVIVING_ITEMS = [
    {"item_id": 1, "frame": 30, "tracks": [2, 8]},
    {"item_id": 2, "frame": 70, "tracks": [3, 4]},
    {"item_id": 3, "frame": 151, "tracks": [25, 26]},
    {"item_id": 4, "frame": 190, "tracks": [49, 58]},
    {"item_id": 5, "frame": 233, "tracks": [71, 72]},
    {"item_id": 6, "frame": 283, "tracks": [70, 95]},
    {"item_id": 7, "frame": 433, "tracks": [136, 141]},
    {"item_id": 8, "frame": 434, "tracks": [73, 114]},
    {"item_id": 9, "frame": 485, "tracks": [126, 150]},
    {"item_id": 10, "frame": 616, "tracks": [192, 196]},
    {"item_id": 11, "frame": 648, "tracks": [192, 201]},
]

TARGET_FRAMES = set(x["frame"] for x in SURVIVING_ITEMS)
MAX_FRAME = max(TARGET_FRAMES)

def main():
    video_path = "C:/Users/KALAKA~1/Downloads/archive/real_videos/5uh8-3q51uI_00.mp4"
    config_path = "road_surveillance/config/default_config.yaml"
    video_id = "5uh8-3q51uI_00"

    cfg = load_config(config_path)

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

    vocab_gate = VocabularyGate()
    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))

    model_path = getattr(cfg.detection, "model_path", "yolov8x.pt")
    yolo_model = YOLO(model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)

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
    motion_gate.reset()

    # Store frame history of track ground positions: frame_idx -> {track_id: (X, Y)}
    frame_ground_positions: Dict[int, Dict[int, Tuple[float, float]]] = {}
    frame_pts_map: Dict[int, float] = {}

    ingestor = DatasetVideoIngestor(path=video_path)

    print(f"Tracking up to frame {MAX_FRAME}...")
    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        if f_idx > MAX_FRAME:
            break

        img = frame.image
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)
        frame_pts_map[f_idx] = timestamp_s

        # 1. Primary YOLO 640p pass
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

        # 2. Motion-Guided ROI Zoom Pass
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

        # 5. Kinematics
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)

        # Cache ground positions for this frame
        frame_ground_positions[f_idx] = {
            tid: k.ground_pos for tid, k in kinematics.items()
        }

    print("Tracking complete. Performing 5-frame kinematic replay analysis...")

    results = []
    for item in SURVIVING_ITEMS:
        item_id = item["item_id"]
        t = item["frame"]
        t1, t2 = item["tracks"]

        # Check availability of [t-4, t-3, t-2, t-1, t]
        history_frames = [t - 4, t - 3, t - 2, t - 1, t]
        
        pos_hist_1 = [frame_ground_positions.get(f, {}).get(t1) for f in history_frames]
        pos_hist_2 = [frame_ground_positions.get(f, {}).get(t2) for f in history_frames]

        dt_total = frame_pts_map[t] - frame_pts_map[t - 4]
        if dt_total <= 0:
            dt_total = 4.0 / 30.0  # fallback 30fps

        p1_t4 = pos_hist_1[0]
        p1_t = pos_hist_1[-1]
        p2_t4 = pos_hist_2[0]
        p2_t = pos_hist_2[-1]

        # 1. 5-frame endpoint velocity & heading vectors:
        v1_bar = np.array([(p1_t[0] - p1_t4[0]) / dt_total, (p1_t[1] - p1_t4[1]) / dt_total], dtype=np.float64)
        v2_bar = np.array([(p2_t[0] - p2_t4[0]) / dt_total, (p2_t[1] - p2_t4[1]) / dt_total], dtype=np.float64)

        spd1_bar = float(np.linalg.norm(v1_bar))
        spd2_bar = float(np.linalg.norm(v2_bar))

        theta1_bar = (math.degrees(math.atan2(v1_bar[1], v1_bar[0])) + 360.0) % 360.0
        theta2_bar = (math.degrees(math.atan2(v2_bar[1], v2_bar[0])) + 360.0) % 360.0

        dot_v = float(np.dot(v1_bar, v2_bar))
        cos_v = dot_v / (spd1_bar * spd2_bar + 1e-6)
        cos_clamped = max(-1.0, min(1.0, cos_v))
        delta_theta_5f = float(np.degrees(np.arccos(cos_clamped)))

        # 2. 5-frame lateral separation dynamics:
        # In this homography, ground coordinate X is the lateral coordinate across lanes
        d_lat_t4 = abs(p1_t4[0] - p2_t4[0])
        d_lat_t = abs(p1_t[0] - p2_t[0])
        delta_d_lat_5f = d_lat_t - d_lat_t4
        is_lat_prox_22 = bool(d_lat_t <= 2.20)

        # 3. Multi-frame linear extrapolation (TCA & DCA) using 5-frame averaged velocities:
        pos1_t = np.array(p1_t, dtype=np.float64)
        pos2_t = np.array(p2_t, dtype=np.float64)
        r_rel = pos2_t - pos1_t
        v_rel = v2_bar - v1_bar
        dist_curr = float(np.linalg.norm(r_rel))
        v_rel_sq = float(np.dot(v_rel, v_rel))

        if v_rel_sq > 1e-4:
            tca_5f = -float(np.dot(r_rel, v_rel)) / v_rel_sq
            if tca_5f > 0.0:
                dca_vec = r_rel + v_rel * tca_5f
                dca_5f = float(np.linalg.norm(dca_vec))
            else:
                dca_5f = dist_curr
        else:
            tca_5f = float("inf")
            dca_5f = dist_curr

        # Evaluate the 4 bounds:
        # Condition 1: Trajectory convergence: delta_theta_5f >= 15.0 deg
        c1 = bool(delta_theta_5f >= 15.0)
        # Condition 2: Compressive lateral closure: delta_d_lat_5f <= -0.25 m
        c2 = bool(delta_d_lat_5f <= -0.25)
        # Condition 3: Lateral proximity: d_lat_t <= 2.20 m
        c3 = bool(d_lat_t <= 2.20)
        # Condition 4: Close approach: TCA <= 1.50 s and DCA <= 1.20 m
        c4 = bool(tca_5f <= 1.50 and dca_5f <= 1.20)

        passes_all_4 = bool(c1 and c2 and c3 and c4)
        outcome = "RETAINED" if passes_all_4 else "SUPPRESSED"

        res = {
            "item_id": item_id,
            "frame": t,
            "tracks": [t1, t2],
            "p1_hist": [[round(float(c), 3) for c in pt] for pt in pos_hist_1],
            "p2_hist": [[round(float(c), 3) for c in pt] for pt in pos_hist_2],
            "v1_bar": [round(float(x), 3) for x in v1_bar],
            "v2_bar": [round(float(x), 3) for x in v2_bar],
            "spd1_kmh": round(spd1_bar * 3.6, 2),
            "spd2_kmh": round(spd2_bar * 3.6, 2),
            "theta1_bar": round(theta1_bar, 2),
            "theta2_bar": round(theta2_bar, 2),
            "delta_theta_5f": round(delta_theta_5f, 2),
            "d_lat_t4": round(d_lat_t4, 3),
            "d_lat_t": round(d_lat_t, 3),
            "delta_d_lat_5f": round(delta_d_lat_5f, 3),
            "is_lat_prox_22": is_lat_prox_22,
            "tca_5f": round(tca_5f, 3) if not math.isinf(tca_5f) else 999.0,
            "dca_5f": round(dca_5f, 3),
            "c1_delta_theta_ge_15": c1,
            "c2_delta_dlat_le_minus_025": c2,
            "c3_dlat_le_22": c3,
            "c4_tca_dca": c4,
            "passes_all_4": passes_all_4,
            "outcome": outcome,
        }
        results.append(res)
        print(f"Item #{item_id} (F{t}, Tracks {t1},{t2}): Δθ={delta_theta_5f:.1f}°, d_lat={d_lat_t4:.2f}→{d_lat_t:.2f}m (Δ={delta_d_lat_5f:.2f}m), TCA={tca_5f:.2f}s, DCA={dca_5f:.2f}m -> {outcome}")

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/offline_5frame_replay_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved offline replay results to artifacts/offline_5frame_replay_results.json")

if __name__ == "__main__":
    main()
