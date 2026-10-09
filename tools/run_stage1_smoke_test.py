"""Stage 1: Daylight Precision Targeted Smoke Test (Iteration R).

Evaluates the 4 primary daytime false-positive scenes:
- BgdKGnFqz0I_00 (GT F125)
- maJGw3hQs_s_00 (GT F31)
- kABWTDKgc9U_00 (GT F197)
- F0NHENn-HS4_00 (GT F39)
"""

import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))
from tools.benchmark_q_full_sweep import run_single_video_evaluation
from road_surveillance.config import load_config
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator
from ultralytics import YOLO

TARGETS = [
    {
        "video_id": "BgdKGnFqz0I_00",
        "gt_frame": 125,
        "split_type": "split_in_distribution",
        "incident_type": "t-bone",
        "day_time": "day",
        "weather": "normal",
    },
    {
        "video_id": "maJGw3hQs_s_00",
        "gt_frame": 31,
        "split_type": "split_in_distribution",
        "incident_type": "t-bone",
        "day_time": "day",
        "weather": "normal",
    },
    {
        "video_id": "kABWTDKgc9U_00",
        "gt_frame": 197,
        "split_type": "split_in_distribution",
        "incident_type": "t-bone",
        "day_time": "day",
        "weather": "normal",
    },
    {
        "video_id": "F0NHENn-HS4_00",
        "gt_frame": 39,
        "split_type": "split_geo_aware",
        "incident_type": "rear-end",
        "day_time": "day",
        "weather": "normal",
    },
]

def main():
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

    print("=" * 105)
    print("STAGE 1: DAYLIGHT PRECISION TARGETED SMOKE TEST (Iteration R)")
    print("Invariants: Native PTS timing | Frozen tau_high=0.25, tau_low=0.10 | Frozen tau_rest >= 0.25s")
    print("=" * 105)
    print(f"{'Video ID':<16} | {'GT F':<5} | {'Status':<6} | {'Raw':<5} | {'Eps':<4} | {'VLM':<4} | {'FP Conf':<7} | {'FP Supp':<7} | {'Trig F':<8} | {'FPS':<5}")
    print("-" * 105)

    results = []
    total_raw = 0
    total_eps = 0
    total_vlm = 0
    total_fp_conf = 0
    total_fp_supp = 0
    all_tp = True

    for t in TARGETS:
        vid = t["video_id"]
        gt_f = t["gt_frame"]
        s_type = t["split_type"]
        vid_path = os.path.join(video_dir, f"{vid}.mp4")

        res = run_single_video_evaluation(
            video_path=vid_path,
            video_id=vid,
            gt_frame=gt_f,
            split_type=s_type,
            weather=t["weather"],
            day_time=t["day_time"],
            yolo_model=yolo_model,
            preprocessor=preprocessor,
            vocab_gate=vocab_gate,
            quality_gate=quality_gate,
            motion_gate=motion_gate,
            arbitrator=arbitrator,
            cfg=cfg,
            tau_single=0.65,
        )

        status_str = "TP" if res["is_tp"] else "FN"
        if not res["is_tp"]:
            all_tp = False

        raw_c = res["raw_candidates_count"]
        eps_c = res["consolidated_episodes_count"]
        vlm_c = res["vlm_calls"]
        fp_conf = res["fp_confirmed_count"]
        fp_supp = res["fp_suppressed_count"]
        trig_f = f"F{res['matched_candidate']['frame']}" if res.get("matched_candidate") else "-"
        fps = res.get("fps", 0.0)

        total_raw += raw_c
        total_eps += eps_c
        total_vlm += vlm_c
        total_fp_conf += fp_conf
        total_fp_supp += fp_supp

        results.append(res)
        print(f"{vid:<16} | F{gt_f:<4} | {status_str:<6} | {raw_c:<5} | {eps_c:<4} | {vlm_c:<4} | {fp_conf:<7} | {fp_supp:<7} | {trig_f:<8} | {fps:<5.1f}")

    print("-" * 105)
    print(f"TOTALS           |       | {'4/4 TP' if all_tp else 'FAIL':<6} | {total_raw:<5} | {total_eps:<4} | {total_vlm:<4} | {total_fp_conf:<7} | {total_fp_supp:<7}")
    print("=" * 105)

    with open("artifacts/stage1_smoke_test.json", "w") as f:
        json.dump({
            "stage": "Stage 1",
            "all_tp": all_tp,
            "total_raw": total_raw,
            "total_eps": total_eps,
            "total_vlm": total_vlm,
            "total_fp_conf": total_fp_conf,
            "total_fp_supp": total_fp_supp,
            "results": results,
        }, f, indent=2)

if __name__ == "__main__":
    main()
