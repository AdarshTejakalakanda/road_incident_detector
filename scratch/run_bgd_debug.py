import os, sys, time
sys.path.insert(0, os.getcwd())
from road_surveillance.config.config_loader import load_config
from tools.benchmark_sa_full_sweep import run_single_video_evaluation
from ultralytics import YOLO
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator

video_dir = r'C:\Users\KALAKA~1\Downloads\archive\real_videos'
vid_path = os.path.join(video_dir, 'BgdKGnFqz0I_00.mp4')
cfg = load_config('road_surveillance/config/default_config.yaml')
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode='N3', low_light_threshold=50.0, gamma=0.65)
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

res = run_single_video_evaluation(
    video_path=vid_path,
    video_id='BgdKGnFqz0I_00',
    gt_frame=125,
    split_type='split_in_distribution',
    weather='normal',
    day_time='day',
    yolo_model=yolo_model,
    preprocessor=preprocessor,
    vocab_gate=vocab_gate,
    quality_gate=quality_gate,
    motion_gate=motion_gate,
    arbitrator=arbitrator,
    cfg=cfg,
    tau_single=0.65,
)

print('Bgd debug result: TP =', res['is_tp'], 'FP =', res['fp_confirmed_count'], 'bypass =', res['bypass_activations'])
print('vlm_tp_result =', res['vlm_tp_result'])
