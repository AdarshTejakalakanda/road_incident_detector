"""Deep extraction script for the 10 surviving records on 5uh8-3q51uI_00.
Extracts continuous bounding boxes, ground positions, speeds, accelerations,
headings, and class labels across [t - 30, t + 30].
Also extracts the full F150 - F240 sequence around the F176 crash.
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

RECORDS = [
    {"id": 1, "frame": 44, "tracks": [4, 5], "path": "B"},
    {"id": 2, "frame": 48, "tracks": [25, 26], "path": "B"},
    {"id": 3, "frame": 196, "tracks": [49, 74], "path": "B"},
    {"id": 4, "frame": 199, "tracks": [74, 77], "path": "A"},
    {"id": 5, "frame": 401, "tracks": [100, 109], "path": "A"},
    {"id": 6, "frame": 446, "tracks": [113, 127], "path": "A"},
    {"id": 7, "frame": 625, "tracks": [169, 188], "path": "A"},
    {"id": 8, "frame": 711, "tracks": [157, 228], "path": "B"},
    {"id": 9, "frame": 712, "tracks": [213, 228], "path": "B"},
    {"id": 10, "frame": 720, "tracks": [196, 201], "path": "A"},
]

def main():
    video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\5uh8-3q51uI_00.mp4"
    config_path = "road_surveillance/config/default_config.yaml"
    cfg = load_config(config_path)

    cap_meta = cv2.VideoCapture(video_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
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

    frame_track_data: Dict[int, Dict[int, Any]] = {}
    frame_pts: Dict[int, float] = {}

    ingestor = DatasetVideoIngestor(path=video_path)
    print("Tracking video up to frame 800...")

    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        if f_idx > 800:
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
                    "bbox": [round(x, 1) for x in tr.current_bbox_xywh],
                    "heading_deg": round(float(k.heading_deg), 1),
                    "speed_kmh": round(float(k.speed_kmh), 1),
                    "accel_ms2": round(float(k.acceleration_mag), 2),
                    "ground_pos": [round(float(p), 2) for p in k.ground_pos],
                    "class_label": tr.class_label,
                    "track_age": tr.track_age,
                    "status": tr.status,
                }
        frame_track_data[f_idx] = current_frame_dict

        if f_idx % 100 == 0:
            print(f"Processed frame {f_idx}/800...")

    # Now let's extract details for the 10 records
    output_records = {}
    for r in RECORDS:
        rid = r["id"]
        f = r["frame"]
        t1, t2 = r["tracks"]
        window_frames = list(range(max(0, f - 30), min(800, f + 31)))

        t1_hist = {w: frame_track_data.get(w, {}).get(t1) for w in window_frames if frame_track_data.get(w, {}).get(t1)}
        t2_hist = {w: frame_track_data.get(w, {}).get(t2) for w in window_frames if frame_track_data.get(w, {}).get(t2)}

        # Also find all other tracks present around f to see scene context
        context_tracks_at_f = list(frame_track_data.get(f, {}).keys())

        output_records[rid] = {
            "record": r,
            "t1_hist": t1_hist,
            "t2_hist": t2_hist,
            "context_tracks_at_f": context_tracks_at_f,
        }

    # Extract detailed sequence around F176 crash (F165 to F215)
    crash_sequence = {}
    for f in range(165, 216):
        crash_sequence[f] = frame_track_data.get(f, {})

    with open("scratch/forensic_10_records_data.json", "w") as f:
        json.dump({
            "records": output_records,
            "crash_sequence": crash_sequence,
        }, f, indent=2)

    print("Data extracted and written to scratch/forensic_10_records_data.json successfully!")

if __name__ == "__main__":
    main()
