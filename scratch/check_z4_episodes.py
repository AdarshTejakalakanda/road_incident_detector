import sys, json
sys.path.insert(0, '.')
from tools.benchmark_q_full_sweep import run_single_video_evaluation
from road_surveillance.config import load_config
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator
from ultralytics import YOLO

cfg = load_config('road_surveillance/config/default_config.yaml')
yolo_model = YOLO(cfg.detection.model_path)
preprocessor = LowLightPreprocessor(mode='N3', low_light_threshold=50.0, gamma=0.65)
vocab_gate = VocabularyGate()
quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = MotionROIGate(crop_resolution=320, crop_conf_threshold=0.05, max_crops_per_frame=1, min_crop_interval=2)
arbitrator = GroqArbitrator(mock_fallback=True)

res = run_single_video_evaluation(
    video_path=r'C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4',
    video_id='Z4kg2Ev3vhk_00',
    gt_frame=170,
    split_type='split_in_distribution',
    weather='normal',
    day_time='night',
    yolo_model=yolo_model,
    preprocessor=preprocessor,
    vocab_gate=vocab_gate,
    quality_gate=quality_gate,
    motion_gate=motion_gate,
    arbitrator=arbitrator,
    cfg=cfg,
)
for ep in res['episodes']:
    print(f"Episode {ep.episode_id}: in_window={ep.is_in_gt_window}, peak={ep.peak_frame}, range=[{ep.start_frame}, {ep.end_frame}], candidates={len(ep.raw_candidates)}, confirmed={ep.is_confirmed}")
    sig = ep.peak_candidate.get('signals', {})
    print(f"   Peak cand: F{ep.peak_candidate['frame']}, type={ep.peak_candidate['event_type']}, tracks={ep.peak_candidate['track_ids']}, accel={sig.get('accel_1_ms2')}, iou={sig.get('bbox_iou')}, dist={sig.get('dist_m')}")
    print(f"   VLM Result: {ep.vlm_result.get('verdict')}, {ep.vlm_result.get('rationale')}")
