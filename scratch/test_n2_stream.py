import sys
sys.path.insert(0, ".")
import cv2
import time
import numpy as np
from ultralytics import YOLO

from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
video_id = "nAXTthLfgtI_00"
gt_frame = 62

cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(
    min_cluster_area=250,
    max_cluster_area=3500,
    persistence_threshold=4,
    crop_resolution=320,
    crop_conf_threshold=0.05,
    max_crops_per_frame=1,
    min_crop_interval=3,
)

cap_meta = cv2.VideoCapture(vpath)
vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
cap_meta.release()

raw_calib = cfg.calibration.get(video_id) or cfg.calibration.get("default")
cam_cfg = CameraConfiguration.from_config_dict(video_id, raw_calib, is_calibrated=False)
lane_mgr = LaneManager.from_config(cam_cfg.lanes, vocab_gate.homography)
geom_engine = GeometryEngine(vocab_gate.homography, lane_mgr, is_camera_calibrated=False)
event_engine = EventEngine(cfg.events, is_camera_calibrated=False)
event_engine.set_resolution(vid_w, vid_h)

road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
for d in event_engine.detectors:
    if isinstance(d, SingleVehicleAnomalyDetector):
        d.tau_single = 0.65
        d.set_roadway_polygon(road_poly)

tracker = RobustSurveillanceTracker()
motion_gate.reset()

active_tracks = {}
all_candidates = []
crops_evaluated = 0
micro_recovered = 0
incident_tracks = {}

ingestor = DatasetVideoIngestor(path=vpath)
t0 = time.perf_counter()
frames_count = 0

for frame in ingestor.stream_video(vpath):
    f_idx = frame.frame_idx
    if f_idx > 200:
        break
    frames_count += 1
    img = frame.image
    proc_img, _ = preprocessor.process(img)
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    # 1. Primary YOLO 640p pass
    yres = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        for bx, bc, bcl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy()):
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": yolo_model.names[int(bcl)].lower(),
            })

    # Active track bounding boxes in xyxy
    active_boxes = []
    for tr in active_tracks.values():
        x, y, w, h = tr.current_bbox_xywh
        active_boxes.append((x, y, x + w, y + h))

    # 2. Motion-Guided ROI Zoom pass
    rec_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=raw_dets,
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
        active_track_boxes=active_boxes,
    )
    raw_dets.extend(rec_micro)
    crops_evaluated += m_tel["roi_crops_evaluated"]
    micro_recovered += m_tel["recovered_micro_vehicles"]

    # 3. Quality Gate
    admitted, _ = quality_gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=f_idx,
        timestamp_s=timestamp_s,
        frame_shape=(vid_h, vid_w),
        homography=vocab_gate.homography,
        roadway_polygon=road_poly,
    )

    # 4. Tracker
    tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

    for tid in active_tracks.keys():
        if abs(f_idx - gt_frame) <= 15:
            incident_tracks[tid] = incident_tracks.get(tid, 0) + 1

    # 5. Kinematics & Event Engine
    kinematics, convergences = geom_engine.process_frame(tracklets, active_tracks)
    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id=video_id,
        tracks=active_tracks,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_mgr,
    )
    for c in cands:
        etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
        all_candidates.append({
            "frame": f_idx,
            "event_type": etype,
            "confidence": float(c.confidence),
            "tracks": c.track_ids,
            "desc": c.description,
        })
        print(f"CANDIDATE F{f_idx}: {etype} conf={c.confidence:.2f} tracks={c.track_ids}")

elapsed = time.perf_counter() - t0
fps = frames_count / elapsed if elapsed > 0 else 0.0

print(f"\nCompleted {frames_count} frames in {elapsed:.2f}s -> {fps:.1f} FPS")
print(f"ROI Crops Evaluated: {crops_evaluated}")
print(f"Micro Vehicles Recovered: {micro_recovered}")
print(f"Incident Tracks: {incident_tracks}")
in_win = [c for c in all_candidates if abs(c["frame"] - gt_frame) <= 35]
print(f"In-Window Candidates near GT F{gt_frame}: {len(in_win)}")
for c in in_win:
    print(f"  -> F{c['frame']} {c['event_type']} conf={c['confidence']} tracks={c['tracks']}")
