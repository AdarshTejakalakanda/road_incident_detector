import os
import sys
import json

sys.path.insert(0, os.path.abspath("."))
from tools.benchmark_q_full_sweep import run_single_video_evaluation
from road_surveillance.config import load_config
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator
from ultralytics import YOLO

video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"
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
arbitrator = GroqArbitrator(mock_fallback=True)

for vid, gt_f in [("UarP8qU1S-c_00", 130), ("ftk_fSc4haM_00", 130)]:
    vid_path = os.path.join(video_dir, f"{vid}.mp4")
    res = run_single_video_evaluation(
        video_path=vid_path,
        video_id=vid,
        gt_frame=gt_f,
        split_type="split_in_distribution",
        weather="normal",
        day_time="night",
        yolo_model=yolo_model,
        preprocessor=preprocessor,
        vocab_gate=vocab_gate,
        quality_gate=quality_gate,
        motion_gate=motion_gate,
        arbitrator=arbitrator,
        cfg=cfg,
    )
    # Print out of window episodes
    print(f"=== {vid} ===")
    print(f"TP: {res['is_tp']}, FP confirmed: {res['fp_confirmed_count']}, FP supp: {res['fp_suppressed_count']}")

