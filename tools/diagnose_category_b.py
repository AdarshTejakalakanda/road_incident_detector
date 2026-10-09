"""Diagnostic script for Category B videos: Z4kg2Ev3vhk_00 and bJMc3sNP2AU_00."""
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

def diagnose_video(vid, gt_frame, split_type, weather, day_time):
    video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"
    vid_path = os.path.join(video_dir, f"{vid}.mp4")
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

    arb_calls = []
    orig_arbitrate = arbitrator.arbitrate
    def debug_arb(evidence_strip, telemetry, candidate_frame=51, video_id="yzNEliTlFGk_00"):
        res = orig_arbitrate(evidence_strip, telemetry, candidate_frame, video_id)
        arb_calls.append({"candidate_frame": candidate_frame, "telemetry": telemetry, "result": res})
        return res
    arbitrator.arbitrate = debug_arb

    res = run_single_video_evaluation(
        video_id=vid,
        video_path=vid_path,
        gt_frame=gt_frame,
        split_type=split_type,
        weather=weather,
        day_time=day_time,
        yolo_model=yolo_model,
        preprocessor=preprocessor,
        vocab_gate=vocab_gate,
        quality_gate=quality_gate,
        motion_gate=motion_gate,
        arbitrator=arbitrator,
        cfg=cfg,
        tau_single=0.65,
    )

    print("=" * 80)
    print(f"DIAGNOSTIC REPORT: {vid} (GT Frame {gt_frame}, Split {split_type})")
    print(f"Status: {'TP' if res['is_tp'] else 'FN'}, Raw Triggers: {res['raw_candidates_count']}, Episodes: {res['consolidated_episodes_count']}, VLM Dispatches: {len(arb_calls)}")
    print("=" * 80)

    for i, call in enumerate(arb_calls, 1):
        f = call["candidate_frame"]
        tel = call["telemetry"]
        r = call["result"]
        print(f"\n--- Episode {i}: Candidate Frame {f} (GT={gt_frame}, Delta={f - gt_frame:+d}f) ---")
        print(f"  Event Type: {tel.get('event_type')}, Anomaly: {tel.get('anomaly_type')}")
        print(f"  Involved Tracks: {tel.get('involved_tracks')}")
        print(f"  Composite Score / Conf: {tel.get('composite_score')}")
        print(f"  Distance (m): {tel.get('distance_m')}, IoU: {tel.get('bbox_iou')}")
        print(f"  Accel (m/s^2): {tel.get('accel_ms2')}, Speed: {tel.get('speed_kmh')} km/h, Max Spd: {tel.get('max_recent_speed_kmh')} km/h")
        print(f"  Yaw Delta: {tel.get('yaw_delta_deg')}, Dist to Edge: {tel.get('d_edge_m')}")
        print(f"  Arbitration Verdict: {r.get('verdict')}")
        print(f"  Suppressed: {r.get('suppressed')}, Reason: {r.get('suppression_reason')}")
        print(f"  Rationale: {r.get('rationale')}")
        if "bypass_telemetry" in r:
            bp = r["bypass_telemetry"]
            print(f"  Bypass Passed: {bp.get('bypass_passed')}, Bypass Reason: {bp.get('bypass_reason')}")

if __name__ == "__main__":
    print("\n>>> DIAGNOSING Z4kg2Ev3vhk_00 <<<")
    diagnose_video("Z4kg2Ev3vhk_00", 170, "split_in_distribution", "normal", "night")

    print("\n\n>>> DIAGNOSING bJMc3sNP2AU_00 <<<")
    diagnose_video("bJMc3sNP2AU_00", 32, "split_geo_aware", "rain", "night")
