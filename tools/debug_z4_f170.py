import os, sys
sys.path.insert(0, os.path.abspath("."))
import cv2
from road_surveillance.config import load_config
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from ultralytics import YOLO

video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(crop_resolution=320, crop_conf_threshold=0.05, max_crops_per_frame=1, min_crop_interval=2)

cap_meta = cv2.VideoCapture(video_path)
vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
cap_meta.release()

raw_calib = cfg.calibration.get("Z4kg2Ev3vhk_00") or cfg.calibration.get("default")
camera_cfg = CameraConfiguration.from_config_dict("Z4kg2Ev3vhk_00", raw_calib, is_calibrated=False)
homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
geometry_engine = GeometryEngine(homography, lane_manager, False)
event_engine = EventEngine(cfg.events, False)
event_engine.set_resolution(vid_w, vid_h)
tracker = RobustSurveillanceTracker()

ingestor = DatasetVideoIngestor(video_path)
for frame in ingestor.stream_video(video_path):
    f_idx = frame.frame_idx
    if f_idx > 185:
        break
    img = frame.image
    proc_img, _ = preprocessor.process(img)
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    yres = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        b_xyxy = yres.boxes.xyxy.cpu().numpy()
        b_conf = yres.boxes.conf.cpu().numpy()
        b_cls = yres.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": yolo_model.names[int(bcl)].lower()
            })

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    recovered_micro, m_tel = motion_gate.process_and_recover(
        frame_curr=img,
        frame_idx=f_idx,
        primary_raw_detections=raw_dets,
        roadway_polygon=road_poly,
        yolo_model=yolo_model,
    )
    raw_dets.extend(recovered_micro)

    admitted_recs, _ = quality_gate.process_and_admit(raw_dets, f_idx, timestamp_s, (vid_h, vid_w), homography, roadway_polygon=road_poly)
    tracklets, track_dict = tracker.update(admitted_recs, pts_ns=pts_ns, frame_idx=f_idx)
    kinematics, convergences = geometry_engine.process_frame(tracklets, track_dict)
    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id="Z4kg2Ev3vhk_00",
        tracks=track_dict,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )
    if 155 <= f_idx <= 175:
        col_det = [d for d in event_engine.detectors if hasattr(d, "suppressed_events")][0]
        for conv in convergences:
            if conv.distance_m <= 25.0:
                t1, t2 = conv.track_id_1, conv.track_id_2
                if t1 in track_dict and t2 in track_dict:
                    trA, trB = track_dict[t1], track_dict[t2]
                    iou = col_det._compute_bbox_iou(trA.current_bbox_xywh, trB.current_bbox_xywh)
                    iou_x = col_det._compute_bbox_iou_x(trA.current_bbox_xywh, trB.current_bbox_xywh)
                    kA, kB = kinematics.get(t1), kinematics.get(t2)
                    accA = kA.acceleration_mag if kA else 0
                    accB = kB.acceleration_mag if kB else 0
                    hdgA = kA.heading_deg if kA else 0
                    hdgB = kB.heading_deg if kB else 0
                    dh = abs(hdgA - hdgB)
                    if dh > 180: dh = 360 - dh
                    pkey = (min(t1, t2), max(t1, t2))
                    c_cnt = col_det.closure_persist_counts.get(pkey, 0)
                    print(f"  F{f_idx} Pair ({t1}, {t2}): dist={conv.distance_m:.1f}m dh={dh:.1f} iou={iou:.3f} iou_x={iou_x:.3f} accs=({accA:.1f}, {accB:.1f}) c_cnt={c_cnt}")
        for c in cands:
            print(f"   CAND {c.event_type} tracks={c.track_ids}")
