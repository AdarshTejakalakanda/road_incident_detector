import sys
sys.path.insert(0, ".")
import os
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
from road_surveillance.evaluation import GroundTruthLoader

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.vlm_arbitrator import GroqArbitrator

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
video_id = "nAXTthLfgtI_00"
gt_frame = 62

cfg = load_config("road_surveillance/config/default_config.yaml")
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
arbitrator = GroqArbitrator(mock_fallback=True)

cap_meta = cv2.VideoCapture(vpath)
vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
cap_meta.release()

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
event_engine = EventEngine(
    config=cfg.events,
    is_camera_calibrated=camera_cfg.is_calibrated,
)
event_engine.set_resolution(vid_w, vid_h)

road_poly = np.array(vocab_gate.get_default_roadway_polygon((vid_h, vid_w)), dtype=np.int32)
roadway_mask = np.zeros((vid_h, vid_w), dtype=np.uint8)
cv2.fillPoly(roadway_mask, [road_poly], 255)

for d in event_engine.detectors:
    if isinstance(d, SingleVehicleAnomalyDetector):
        d.tau_single = 0.65
        d.set_roadway_polygon(road_poly.tolist())

tracker = RobustSurveillanceTracker()

tracked_clusters = []
next_cluster_id = 1
prev_frame = None

def compute_box_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

all_candidates = []
ingestor = DatasetVideoIngestor(path=vpath)
t0 = time.perf_counter()
frames_processed = 0

for frame in ingestor.stream_video(vpath):
    f_idx = frame.frame_idx
    if f_idx > 200:
        break
    frames_processed += 1
    img = frame.image
    proc_img, _ = preprocessor.process(img)
    pts_ns = frame.pts_ns
    timestamp_s = float(pts_ns * 1e-9)

    # 1. Primary YOLO Pass at 640p
    yolo_res = yolo_model.predict(source=proc_img, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    raw_dets = []
    primary_veh_boxes = []

    if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
        b_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
        b_conf = yolo_res.boxes.conf.cpu().numpy()
        b_cls = yolo_res.boxes.cls.cpu().numpy()
        for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
            c_name = yolo_model.names[int(bcl)].lower()
            raw_dets.append({
                "bbox": [float(x) for x in bx],
                "conf": float(bc),
                "class_id": int(bcl),
                "class_name": c_name,
            })
            if c_name in ("car", "truck", "bus", "motorcycle", "van", "train"):
                primary_veh_boxes.append(bx)

    # 2. Motion Differencing & Cluster Tracking
    if prev_frame is not None:
        diff = cv2.absdiff(img, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=roadway_mask)
        _, thresh = cv2.threshold(masked_diff, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_rois = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 250 <= area <= 3500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                pad_x, pad_y = int(bw * 0.5), int(bh * 0.5)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(vid_w, x + bw + pad_x)
                y2 = min(vid_h, y + bh + pad_y)
                current_rois.append((x1, y1, x2, y2, area))

        # Match with tracked clusters
        matched_ids = set()
        for roi in current_rois:
            rbox = roi[:4]
            rcx, rcy = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2
            best_match = None
            best_dist = 80.0
            for tc in tracked_clusters:
                if tc["id"] in matched_ids:
                    continue
                tcx, tcy = (tc["box"][0] + tc["box"][2]) / 2, (tc["box"][1] + tc["box"][3]) / 2
                d = np.hypot(rcx - tcx, rcy - tcy)
                if d < best_dist:
                    best_dist = d
                    best_match = tc
            if best_match:
                best_match["box"] = rbox
                best_match["age"] += 1
                best_match["last_frame"] = f_idx
                best_match["area"] = roi[4]
                matched_ids.add(best_match["id"])
            else:
                tracked_clusters.append({
                    "id": next_cluster_id,
                    "box": rbox,
                    "age": 1,
                    "last_frame": f_idx,
                    "area": roi[4],
                })
                next_cluster_id += 1

        tracked_clusters = [tc for tc in tracked_clusters if f_idx - tc["last_frame"] <= 2]

        # Invocation Condition: cluster age >= 4, no overlapping primary vehicle
        candidate_clusters = [
            tc for tc in tracked_clusters
            if tc["age"] >= 4 and not any(compute_box_iou(tc["box"], pb) > 0.05 for pb in primary_veh_boxes)
        ]

        if candidate_clusters:
            # Enforce maximum of 1 ROI crop per frame
            best_cluster = max(candidate_clusters, key=lambda c: (c["age"], c["area"]))
            bx1, by1, bx2, by2 = best_cluster["box"]
            if (bx2 - bx1) >= 15 and (by2 - by1) >= 15:
                crop = img[by1:by2, bx1:bx2]
                crop_320 = cv2.resize(crop, (320, 320))
                zres = yolo_model.predict(source=crop_320, imgsz=320, conf=0.05, device="cpu", verbose=False)[0]
                if zres.boxes is not None and len(zres.boxes) > 0:
                    for zb, zc, zcl in zip(zres.boxes.xyxy.cpu().numpy(), zres.boxes.conf.cpu().numpy(), zres.boxes.cls.cpu().numpy()):
                        zname = yolo_model.names[int(zcl)].lower()
                        if zname == "train":
                            zname = "truck"
                        if zname in ("car", "truck", "bus", "motorcycle"):
                            sx = (bx2 - bx1) / 320.0
                            sy = (by2 - by1) / 320.0
                            gx1 = float(bx1 + zb[0] * sx)
                            gy1 = float(by1 + zb[1] * sy)
                            gx2 = float(bx1 + zb[2] * sx)
                            gy2 = float(by1 + zb[3] * sy)
                            raw_dets.append({
                                "bbox": [gx1, gy1, gx2, gy2],
                                "conf": float(zc),
                                "class_id": int(zcl),
                                "class_name": "candidate_vehicle_recovered",
                                "original_class_name": zname,
                                "recovery_reason": "motion_guided_roi_zoom",
                                "audit_reclassified": True,
                            })
                            print(f"F{f_idx}: Motion ROI zoom recovered {zname}:{zc:.2f} at [{gx1:.1f},{gy1:.1f},{gx2:.1f},{gy2:.1f}]")

    prev_frame = img

    # 3. Quality Gate
    admitted, qtel = quality_gate.process_and_admit(
        raw_detections=raw_dets,
        frame_idx=f_idx,
        timestamp_s=timestamp_s,
        frame_shape=(vid_h, vid_w),
        homography=homography,
        roadway_polygon=road_poly.tolist(),
    )

    # 4. Tracker
    tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

    # 5. Kinematics & Events
    kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
    cands = event_engine.process_frame(
        frame_idx=f_idx,
        pts_ns=pts_ns,
        video_id=video_id,
        tracks=active_tracks,
        kinematics=kinematics,
        convergences=convergences,
        lane_manager=lane_manager,
    )
    for c in cands:
        etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
        all_candidates.append({
            "frame": f_idx,
            "timestamp_s": round(timestamp_s, 3),
            "event_type": etype,
            "track_ids": c.track_ids,
            "confidence": round(float(c.confidence), 2),
            "description": c.description,
            "signals": c.signals or {},
        })
        print(f"  >>> CANDIDATE F{f_idx}: {etype} conf={c.confidence:.2f} tracks={c.track_ids} desc={c.description}")

elapsed = time.perf_counter() - t0
fps = frames_processed / elapsed if elapsed > 0 else 0
print(f"\nFinished: {frames_processed} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
print(f"Total candidates triggered: {len(all_candidates)}")
in_win = [c for c in all_candidates if abs(c["frame"] - gt_frame) <= 35]
print(f"In-window candidates near GT F{gt_frame}: {len(in_win)}")
for c in in_win:
    print(f"  -> In-window F{c['frame']}: {c['event_type']} conf={c['confidence']} tracks={c['track_ids']}")
