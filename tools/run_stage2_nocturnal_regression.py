"""Stage 2: Nocturnal Control Suite Regression (Iteration R).

Runs the 10 frozen nocturnal benchmark evaluation runs:
- Z4kg2Ev3vhk_00 (split_in_distribution, F170)
- unS0-TLF1ao_00 (split_in_distribution, F64)
- nAXTthLfgtI_00 (split_in_distribution, F62)
- OXVXsdDT9QQ_00 (split_in_distribution, F17)
- ftk_fSc4haM_00 (split_in_distribution, F130)
- Z4kg2Ev3vhk_00 (split_geo_aware, F170)
- UarP8qU1S-c_00 (split_geo_aware, F130)
- UarP8qU1S-c_01 (split_geo_aware, F30)
- nAXTthLfgtI_00 (split_geo_aware, F62)
- bJMc3sNP2AU_00 (split_geo_aware, F32)

Gate 2 (Hard Gate): Exactly 0 confirmed nocturnal False Positives.
Verify that ftk_fSc4haM_00 remains confirmed True Positive at F101.
"""

import os
import sys
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

NOCTURNAL_TARGETS = [
    {"video_id": "Z4kg2Ev3vhk_00", "split_type": "split_in_distribution", "gt_frame": 170, "incident_type": "rear-end", "weather": "normal"},
    {"video_id": "unS0-TLF1ao_00", "split_type": "split_in_distribution", "gt_frame": 64, "incident_type": "t-bone", "weather": "normal"},
    {"video_id": "nAXTthLfgtI_00", "split_type": "split_in_distribution", "gt_frame": 62, "incident_type": "single", "weather": "normal"},
    {"video_id": "OXVXsdDT9QQ_00", "split_type": "split_in_distribution", "gt_frame": 17, "incident_type": "t-bone", "weather": "normal"},
    {"video_id": "ftk_fSc4haM_00", "split_type": "split_in_distribution", "gt_frame": 130, "incident_type": "t-bone", "weather": "normal"},
    {"video_id": "Z4kg2Ev3vhk_00", "split_type": "split_geo_aware", "gt_frame": 170, "incident_type": "rear-end", "weather": "normal"},
    {"video_id": "UarP8qU1S-c_00", "split_type": "split_geo_aware", "gt_frame": 130, "incident_type": "single", "weather": "normal"},
    {"video_id": "UarP8qU1S-c_01", "split_geo_aware": "split_geo_aware", "split_type": "split_geo_aware", "gt_frame": 30, "incident_type": "single", "weather": "normal"},
    {"video_id": "nAXTthLfgtI_00", "split_type": "split_geo_aware", "gt_frame": 62, "incident_type": "single", "weather": "normal"},
    {"video_id": "bJMc3sNP2AU_00", "split_type": "split_geo_aware", "gt_frame": 32, "incident_type": "t-bone", "weather": "rain"},
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

    print("=" * 115)
    print("STAGE 2: NOCTURNAL CONTROL SUITE REGRESSION (10 Videos) - Iteration R")
    print("Hard Invariant: Exactly 0 confirmed nocturnal False Positives")
    print("=" * 115)
    print(f"{'#':<2} | {'Video ID':<16} | {'Split':<22} | {'GT F':<5} | {'Type':<8} | {'Status':<6} | {'Raw':<4} | {'Eps':<4} | {'VLM':<4} | {'FP':<4} | {'Trig F':<8} | {'FPS':<5}")
    print("-" * 115)

    results = []
    tp_count = 0
    fn_count = 0
    total_fp = 0
    ftk_recovered = False
    ftk_trig = None

    for i, t in enumerate(NOCTURNAL_TARGETS, 1):
        vid = t["video_id"]
        gt_f = t["gt_frame"]
        s_type = t["split_type"]
        itype = t["incident_type"]
        weather = t["weather"]
        vid_path = os.path.join(video_dir, f"{vid}.mp4")

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
            tau_single=0.65,
        )

        status_str = "TP" if res["is_tp"] else "FN"
        if res["is_tp"]:
            tp_count += 1
        else:
            fn_count += 1

        raw_c = res["raw_candidates_count"]
        eps_c = res["consolidated_episodes_count"]
        vlm_c = res["vlm_calls"]
        fp_c = res["fp_confirmed_count"]
        total_fp += fp_c
        trig_f = f"F{res['matched_candidate']['frame']}" if res.get("matched_candidate") else "-"
        fps = res.get("fps", 0.0)

        if vid == "ftk_fSc4haM_00":
            ftk_recovered = res["is_tp"]
            ftk_trig = trig_f

        results.append(res)
        print(f"{i:02d} | {vid:<16} | {s_type:<22} | F{gt_f:<4} | {itype:<8} | {status_str:<6} | {raw_c:<4} | {eps_c:<4} | {vlm_c:<4} | {fp_c:<4} | {trig_f:<8} | {fps:<5.1f}")

    print("-" * 115)
    print(f"NOCTURNAL SUMMARY: TP={tp_count}/10 | FN={fn_count} | CONFIRMED FP={total_fp}")
    print(f"ftk_fSc4haM_00 Status: {'RECOVERED TP (' + str(ftk_trig) + ')' if ftk_recovered else 'FAILED (FN)'}")
    print(f"Gate 2 Hard Specificity Check: {'PASSED (0 Nocturnal FP)' if total_fp == 0 else f'FAILED ({total_fp} Nocturnal FP)'}")
    print("=" * 115)

    with open("artifacts/stage2_nocturnal_regression.json", "w") as f:
        json.dump({
            "stage": "Stage 2",
            "tp_count": tp_count,
            "fn_count": fn_count,
            "total_nocturnal_fp": total_fp,
            "ftk_recovered": ftk_recovered,
            "ftk_trigger_frame": ftk_trig,
            "gate2_passed": total_fp == 0 and ftk_recovered,
            "results": results,
        }, f, indent=2, default=str)

if __name__ == "__main__":
    main()
