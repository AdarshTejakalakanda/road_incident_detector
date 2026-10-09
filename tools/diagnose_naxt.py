"""Diagnostic trace for nAXTthLfgtI_00 across Gates 1 to 5 for frames 45-65."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from road_surveillance.config import load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from ultralytics import YOLO

def main():
    video_path = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
    video_id = "nAXTthLfgtI_00"
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
        min_crop_interval=2,
    )
    tracker = RobustSurveillanceTracker(
        track_activation_threshold=0.10,
        low_confidence_floor=0.06,
        lost_track_buffer=cfg.tracking.lost_track_buffer,
        match_threshold_high=0.75,
        match_threshold_low=0.65,
    )

    cam_entry = cfg.calibration.get(video_id, cfg.calibration.get("default"))
    camera_cfg = CameraConfiguration.from_config_dict(video_id, cam_entry, False)
    homography = PlanarHomography.from_camera_config(camera_cfg, (960, 720))
    lane_mgr = LaneManager.from_config(camera_cfg.lanes, homography)
    geo_engine = GeometryEngine(homography=homography, lane_manager=lane_mgr)
    sv_detector = SingleVehicleAnomalyDetector(
        is_camera_calibrated=False,
        tau_single=0.65,
        roadway_polygon=getattr(lane_mgr, "roadway_polygon", None),
    )

    ingestor = DatasetVideoIngestor(video_path)

    print("=" * 100)
    print("STEP 1: UPSTREAM DIAGNOSTIC TRACE FOR nAXTthLfgtI_00 (Frames 45-65)")
    print("=" * 100)

    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        img = frame.image
        h, w = img.shape[:2]
        pts_ns = frame.pts_ns
        ts_s = pts_ns * 1e-9 if pts_ns > 0 else f_idx / 15.0

        proc_img, _ = preprocessor.process(img)

        # Gate 1: Raw YOLO Detections
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

        # Motion Gate
        rec_micro, _ = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=getattr(lane_mgr, "roadway_polygon", None),
            yolo_model=yolo_model,
        )
        raw_dets.extend(rec_micro)

        # Gate 2: Detection Quality Gate
        admitted, qtel = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=ts_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=getattr(lane_mgr, "roadway_polygon", None),
        )

        # Gate 3: Tracker Update
        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

        # Kinematics
        kinematics, convergences = geo_engine.process_frame(tracklets, active_tracks)

        # Gate 5: Candidate Generator
        cands = sv_detector.evaluate(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=active_tracks,
            kinematics=kinematics,
            convergences=convergences,
            lane_manager=lane_mgr,
        )

        if 45 <= f_idx <= 65:
            print(f"\n--- FRAME {f_idx} (t={ts_s:.3f}s) ---")
            # Gate 1:
            raw_vehicles = [d for d in raw_dets if d["class_name"] in ("car", "truck", "bus", "motorcycle", "vehicle")]
            print(f"Gate 1 (Raw YOLO >=0.02): {len(raw_dets)} total, {len(raw_vehicles)} vehicles")
            for rv in raw_vehicles:
                bx = [round(x, 1) for x in rv["bbox"]]
                print(f"  Raw: cls={rv['class_name']}, conf={rv['conf']:.3f}, bbox={bx}")

            # Gate 2:
            print(f"Gate 2 (Admitted): {len(admitted)} admitted detections")
            for adm in admitted:
                bx = [round(x, 1) for x in adm.bbox_xyxy]
                print(f"  Admitted: fused_cls={adm.fused_class}, conf={adm.confidence:.3f}, trust={adm.trust_weight:.2f}, bbox={bx}")

            # Gate 3 & 4:
            print(f"Gate 3 & 4 (Active Tracks): {len(active_tracks)} tracks")
            for tid, tr in active_tracks.items():
                k = kinematics.get(tid)
                spd = k.speed_kmh if k else 0.0
                acc = k.acceleration_mag if k else 0.0
                pre_spd = tr.get_pre_impulse_speed(window_s=1.0)
                bx = [round(x, 1) for x in tr.current_bbox_xywh]
                print(f"  Track {tid}: age={tr.track_age}, conf={tr.detection_confidence:.2f}, spd={spd:.1f}km/h, pre_spd={pre_spd:.1f}km/h, acc={acc:.1f}, bbox_xywh={bx}")

            # Gate 5:
            print(f"Gate 5 (Candidates): {len(cands)} candidates triggered")
            for c in cands:
                print(f"  TRIGGER: type={c.event_type}, tracks={c.track_ids}, conf={c.confidence:.2f}, signals={c.signals}")

            # Also check if any suppression occurred in sv_detector
            if sv_detector.suppressed_events:
                curr_supp = [s for s in sv_detector.suppressed_events if s.get("frame") == f_idx]
                if curr_supp:
                    print(f"  Suppressed in SingleVehicleDetector: {len(curr_supp)} events")
                    for s in curr_supp:
                        print(f"    Suppressed: tracks={s['tracks']}, reason={s['reason']}, metric={s.get('distance_or_dca')}")

        if f_idx > 65:
            break

if __name__ == "__main__":
    main()
