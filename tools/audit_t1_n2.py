"""Subtask T.1: Diagnostic Audit of N.2 Motion-Guided ROI Recovery (nAXTthLfgtI_00) across frames 45-65."""
import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from road_surveillance.config import load_config
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate, extract_roadway_motion_clusters
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from ultralytics import YOLO

def audit_t1():
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
        crop_conf_threshold=0.01,  # Lower to 0.01 to see ALL raw detector logits/confidences prior to filtering
        max_crops_per_frame=1,
        min_crop_interval=1,
    )

    cam_entry = cfg.calibration.get(video_id, cfg.calibration.get("default"))
    camera_cfg = CameraConfiguration.from_config_dict(video_id, cam_entry, False)
    homography = PlanarHomography.from_camera_config(camera_cfg, (960, 720))
    lane_mgr = LaneManager.from_config(camera_cfg.lanes, homography)
    road_poly = getattr(lane_mgr, "roadway_polygon", None)

    ingestor = DatasetVideoIngestor(video_path)

    print("=" * 100)
    print("SUBTASK T.1: DIAGNOSTIC AUDIT OF N.2 MOTION-GUIDED ROI RECOVERY (nAXTthLfgtI_00)")
    print("Frames 45 to 65")
    print("=" * 100)

    prev_frame_raw = None

    for frame in ingestor.stream_video(video_path):
        f_idx = frame.frame_idx
        img = frame.image
        h, w = img.shape[:2]

        proc_img, _ = preprocessor.process(img)

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

        # Frame difference / motion energy analysis
        roadway_mask = motion_gate._ensure_roadway_mask((h, w), road_poly)
        diff_max = 0.0
        diff_mean = 0.0
        cluster_list = []
        if prev_frame_raw is not None:
            diff = cv2.absdiff(img, prev_frame_raw)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=roadway_mask)
            diff_max = float(np.max(masked_diff))
            diff_mean = float(np.mean(masked_diff))
            cluster_list = extract_roadway_motion_clusters(
                frame_curr=img,
                frame_prev=prev_frame_raw,
                roadway_mask=roadway_mask,
                min_cluster_area=motion_gate.min_cluster_area,
                max_cluster_area=motion_gate.max_cluster_area,
            )

        # Motion Gate process
        rec_micro, m_tel = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )

        if 45 <= f_idx <= 65:
            print(f"\n[FRAME {f_idx}]")
            print(f"  Frame Differencing (epsilon_motion=25): max_diff={diff_max:.1f}, mean_diff={diff_mean:.2f}")
            print(f"  Motion Clusters Found: {len(cluster_list)}")
            for c in cluster_list:
                print(f"    Raw Cluster: box={[c[0], c[1], c[2], c[3]]}, area={c[4]:.1f}")
            print(f"  Tracked Clusters in Gate: {len(motion_gate._tracked_clusters)}")
            for tc in motion_gate._tracked_clusters:
                print(f"    Tracked ID {tc['id']}: age={tc['age']}, box={tc['box']}, area={tc['area']:.1f}, hits={tc.get('active_hits', 0)}")
            print(f"  ROI Evaluated: {m_tel['roi_crops_evaluated']}, Recovered: {m_tel['recovered_micro_vehicles']}")
            for rm in rec_micro:
                print(f"    RECOVERED: orig_cls={rm['original_class_name']}, conf={rm['conf']:.3f}, bbox={[round(x, 1) for x in rm['bbox']]}")

            # If motion cluster existed, test raw YOLO on crop directly to report logits down to 0.001
            for c in cluster_list:
                bx1, by1, bx2, by2 = c[:4]
                if (bx2 - bx1) >= 15 and (by2 - by1) >= 15:
                    crop = img[by1:by2, bx1:bx2]
                    crop_scaled = cv2.resize(crop, (320, 320))
                    crop_res = yolo_model.predict(source=crop_scaled, imgsz=320, conf=0.005, device="cpu", verbose=False)[0]
                    if crop_res.boxes is not None and len(crop_res.boxes) > 0:
                        print(f"    --> Raw YOLO on Crop [{bx1}, {by1}, {bx2}, {by2}]:")
                        for zb, zc, zcl in zip(crop_res.boxes.xyxy.cpu().numpy(), crop_res.boxes.conf.cpu().numpy(), crop_res.boxes.cls.cpu().numpy()):
                            print(f"        cls={yolo_model.names[int(zcl)]}, conf={float(zc):.4f}, box={[round(float(x), 1) for x in zb]}")
                    else:
                        print(f"    --> Raw YOLO on Crop [{bx1}, {by1}, {bx2}, {by2}]: No detections even at conf=0.005")

        prev_frame_raw = img.copy()

        if f_idx > 65:
            break

if __name__ == "__main__":
    audit_t1()
