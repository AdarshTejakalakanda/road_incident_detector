import os
import sys
import json
import time
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath("."))
from tools.benchmark_q_full_sweep import run_single_video_evaluation
from road_surveillance.config import load_config
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from modules.vocabulary_gate import VocabularyGate
from modules.detector_quality_gate import DetectorQualityGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import GroqArbitrator, passes_tier1_admission
from road_surveillance.evaluation import GroundTruthLoader
from ultralytics import YOLO

class InterceptingArbitrator(GroqArbitrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quarantined_telemetries: List[Dict[str, Any]] = []

    def arbitrate(self, evidence_strip, telemetry, candidate_frame=51, video_id='unknown', mock_vlm_response=None):
        res = super().arbitrate(evidence_strip, telemetry, candidate_frame, video_id, mock_vlm_response)
        if res.get("verdict") == "CANDIDATE_UNCONFIRMED":
            # Save copy of telemetry
            t_copy = dict(telemetry)
            t_copy["_arbitration_result"] = res
            t_copy["_candidate_frame"] = candidate_frame
            t_copy["_video_id"] = video_id
            self.quarantined_telemetries.append(t_copy)
        return res

def main():
    metadata_path = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv"
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

    arbitrator = InterceptingArbitrator(mock_fallback=True)
    loader = GroundTruthLoader(metadata_path)

    # Load 30 benchmark sweep entries
    from tools.benchmark_k0_detector import load_30_benchmark_entries
    benchmark_entries = load_30_benchmark_entries(metadata_path)

    print(f"Running audit across {len(benchmark_entries)} benchmark entries to capture all quarantined fallbacks...")
    detection_cache = {}

    for idx, entry in enumerate(benchmark_entries, 1):
        vid = entry["video_id"]
        gt_f = int(entry["gt_frame"])
        s_type = entry["split_type"]
        gt_rec = loader.get_by_video_id(vid)
        weather = gt_rec.weather if gt_rec else entry.get("weather", "normal")
        day_time = gt_rec.day_time if gt_rec else entry.get("day_time", "day")
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            continue

        run_single_video_evaluation(
            video_path=vpath,
            video_id=vid,
            gt_frame=gt_f,
            split_type=s_type,
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
            detection_cache=detection_cache,
        )

    quar_list = arbitrator.quarantined_telemetries
    print(f"\nTotal quarantined candidates captured: {len(quar_list)}")

    # Programmatic evaluation against Corrected Tier 1 Filter
    surviving_proposals = []
    rejected_proposals = []

    for i, t in enumerate(quar_list, 1):
        passes, reason = passes_tier1_admission(t)
        cand_record = {
            "index": i,
            "candidate_id": f"QUAR-AUDIT-{i:02d}",
            "video_id": t["_video_id"],
            "frame": t["_candidate_frame"],
            "event_type": t.get("event_type"),
            "accel_ms2": t.get("accel_ms2"),
            "speed_kmh": t.get("speed_kmh"),
            "max_recent_speed_kmh": t.get("max_recent_speed_kmh"),
            "delta_v_drop": t.get("delta_v_drop", max(0.0, float(t.get("max_recent_speed_kmh", 0.0)) - float(t.get("speed_kmh", 0.0)))),
            "tau_rest": t.get("tau_rest", 0.25 if t.get("post_event_rest") else 0.0),
            "distance_m": t.get("distance_m"),
            "norm_pixel_dist": t.get("norm_pixel_dist"),
            "bbox_iou": t.get("bbox_iou"),
            "d_edge_m": t.get("d_edge_m"),
            "v_dot_n": t.get("v_dot_n"),
            "passes_tier1": passes,
            "tier1_reason": reason,
        }
        if passes:
            surviving_proposals.append(cand_record)
        else:
            rejected_proposals.append(cand_record)

    print(f"Corrected Tier 1 Kinematic Filter Results:")
    print(f"  Total Evaluated: {len(quar_list)}")
    print(f"  Surviving Proposals (N_surviving): {len(surviving_proposals)}")
    print(f"  Rejected Proposals: {len(rejected_proposals)}")

    out_data = {
        "total_quarantined_captured": len(quar_list),
        "surviving_count": len(surviving_proposals),
        "rejected_count": len(rejected_proposals),
        "surviving_proposals": surviving_proposals,
        "rejected_proposals": rejected_proposals,
    }

    with open("scratch/quarantined_70_audit.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print("Saved results to scratch/quarantined_70_audit.json")

if __name__ == "__main__":
    main()
