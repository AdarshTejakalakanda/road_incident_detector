import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from tools.benchmark_q_full_sweep import run_single_video_evaluation
from road_surveillance.config import load_config
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator
from ultralytics import YOLO

def main():
    cfg = load_config("road_surveillance/config/default_config.yaml")
    yolo_model = YOLO(cfg.detection.model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    vocab_gate = VocabularyGate()
    quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
    motion_gate = MotionROIGate(
        min_cluster_area=250, max_cluster_area=3500, persistence_threshold=4,
        crop_resolution=320, crop_conf_threshold=0.05, max_crops_per_frame=1, min_crop_interval=2
    )
    arbitrator = GroqArbitrator(mock_fallback=True)

    targets = [
        {"video_id": "nAXTthLfgtI_00", "gt_frame": 62, "split_type": "split_in_distribution", "weather": "normal"},
        {"video_id": "Z4kg2Ev3vhk_00", "gt_frame": 170, "split_type": "split_in_distribution", "weather": "normal"},
        {"video_id": "bJMc3sNP2AU_00", "gt_frame": 32, "split_type": "split_geo_aware", "weather": "rain"},
    ]

    for t in targets:
        vid = t["video_id"]
        gt_f = t["gt_frame"]
        s_type = t["split_type"]
        weather = t["weather"]
        vid_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", f"{vid}.mp4")

        print("=" * 80)
        print(f"ANALYZING TARGET: {vid} (GT Frame {gt_f}, Weather: {weather}, Split: {s_type})")
        print("=" * 80)

        res = run_single_video_evaluation(
            video_path=vid_path,
            video_id=vid,
            gt_frame=gt_f,
            split_type=s_type,
            weather=weather,
            day_time="night",
            yolo_model=yolo_model,
            preprocessor=preprocessor,
            vocab_gate=vocab_gate,
            quality_gate=quality_gate,
            motion_gate=motion_gate,
            arbitrator=arbitrator,
            cfg=cfg,
        )

        print(f"Result is_tp: {res['is_tp']}")
        print(f"Raw Candidates Count: {res['raw_candidates_count']}")
        print(f"Consolidated Episodes: {res['consolidated_episodes_count']}")
        print(f"In-Window Episodes: {res['in_window_episodes_count']}")
        print(f"Out-of-Window Episodes: {res['out_of_window_episodes_count']}")
        print(f"Confirmed FP: {res['fp_confirmed_count']}")
        print(f"Failure Gate: {res.get('failure_gate')}")
        print("\nAll Raw Candidates:")
        for c in res["all_candidates"]:
            f = c["frame"]
            ts = c["timestamp_s"]
            etype = c["event_type"]
            conf = c["confidence"]
            tids = c["track_ids"]
            sig = c.get("signals", {})
            print(f"  Frame {f:3d} ({ts:.3f}s) | {etype:<25} | tracks={tids} | conf={conf:.2f} | accel={sig.get('accel_1_ms2', sig.get('accel_ms2', 0.0)):.1f} | dist={sig.get('dist_m', 0.0):.2f} | iou={sig.get('bbox_iou', 0.0):.2f} | closure_corridor={sig.get('is_kinematic_closure_corridor', False)}")

        print("\nConsolidated Episodes Details:")
        for ep in res.get("episodes", []):
            ts_peak = ep.peak_candidate.get('timestamp_s', 0.0) if ep.peak_candidate else 0.0
            print(f"  Episode {ep.episode_id} | In-Window: {ep.is_in_gt_window} | Type: {ep.primary_event_type} | Frame Range: [{ep.start_frame}, {ep.end_frame}] | Peak: F{ep.peak_frame} ({ts_peak:.3f}s)")
            print(f"    Raw Candidate Count: {len(ep.raw_candidates)}")
            print(f"    Peak Candidate Signals: {ep.peak_candidate.get('signals', {})}")
            print(f"    VLM Dispatched: {ep.vlm_dispatched}")
            print(f"    VLM Result: {ep.vlm_result}")
            print(f"    Is Confirmed: {ep.is_confirmed}")

        print("\nMatched Candidate (TP):")
        print(json.dumps(res.get("matched_candidate"), indent=2, default=str))

        print("\nVLM TP Result:")
        print(json.dumps(res.get("vlm_tp_result"), indent=2, default=str))

if __name__ == "__main__":
    main()
