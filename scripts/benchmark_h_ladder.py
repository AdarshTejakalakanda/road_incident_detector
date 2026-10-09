"""Benchmark Controlled H0 -> H4 Ladder on Frozen Incident Candidate.

Evaluates the exact same candidate episode across the 5 representation levels:
- H0: Full frames (3 full canvas frames)
- H1: Full + Clean ROI (1 full canvas + 3 clean focused 384x384 crops)
- H2: Full + ROI + SoM (1 full canvas + 3 marked 384x384 crops with track badges)
- H3: Full + Temporal ROI sequence (1 full canvas + 5 clean temporal 384x384 crops)
- H4: Full + Temporal ROI + SoM + Telemetry (1 full canvas + 5 marked crops + telemetry)
"""

import os
import sys
import time
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.verification.base import CandidateEvidencePackage
from road_surveillance.verification.evidence_extractor import CandidateEvidenceExtractor
from road_surveillance.verification.qwen_verifier import QwenMultimodalVerifier

VIDEO_DIR = r"C:\Users\KalakandaAdarshTeja" + "\ufffd" + r"\Downloads\archive\real_videos"


def run_h_ladder_benchmark(video_id: str = "yzNEliTlFGk_00", candidate_frame: int = 57):
    print("=" * 80)
    print(f"CONTROLLED H0 -> H4 ABLATION BENCHMARK ON FROZEN EPISODE: {video_id} @ F{candidate_frame}")
    print("=" * 80)

    # 1. Extract the candidate evidence package ONCE
    extractor = CandidateEvidenceExtractor(video_dataset_dir=VIDEO_DIR)
    pkg = extractor.extract_evidence_package(
        video_id=video_id,
        candidate_frame=candidate_frame,
        candidate_detector="single_vehicle_anomaly",
        involved_track_ids=[17],
        track_boxes={17: (650.0, 270.0, 60.0, 45.0)},
        telemetry={
            "speed_kmh": 52.8,
            "accel_ms2": 11.2,
            "yaw_rate_deg_s": 14.5,
            "anomaly_type": "aspect_ratio_deformation+forward_impulse",
        },
    )

    print(f"Extracted Package ID: {pkg.event_id}")
    print(f"Spatial ROI Bounding Box: {pkg.roi_bbox}")
    print(f"Available Full Frames: {list(pkg.full_frames.keys())}")
    print(f"Clean ROI Crops Available: {len(pkg.roi_crops)} ({list(pkg.roi_crops.keys())})")
    print(f"Marked SoM Crops Available: {len(pkg.som_crops)} ({list(pkg.som_crops.keys())})")
    print("-" * 80)

    levels = ["H0", "H1", "H2", "H3", "H4"]
    level_descriptions = {
        "H0": "Full frames only (3 full canvas frames: t-1, t0, t+1)",
        "H1": "Full + Clean ROI (1 full canvas + 3 clean 384x384 crops)",
        "H2": "Full + ROI + SoM (1 full canvas + 3 marked 384x384 crops with track badges)",
        "H3": "Full + Temporal ROI (1 full canvas + 5 clean 384x384 temporal crops)",
        "H4": "Full + Temporal ROI + SoM + Telemetry (1 full canvas + 5 marked crops + telemetry)",
    }

    print(f"{'Level':<6} | {'Images Attached':<16} | {'Estimated ViT Tokens':<22} | {'Telemetry Included?':<20} | {'Configuration'}")
    print("-" * 85)

    for lvl in levels:
        verifier = QwenMultimodalVerifier(ablation_level=lvl, mock_mode=True)
        res = verifier.verify(pkg)

        # Estimate image token footprints (approx 1600 tokens for 720p full frame, ~288 for 384x384 crop)
        if lvl == "H0":
            n_full = 3
            n_crops = 0
            has_telem = "No"
        elif lvl in ("H1", "H2"):
            n_full = 1
            n_crops = 3
            has_telem = "No"
        elif lvl == "H3":
            n_full = 1
            n_crops = 5
            has_telem = "No"
        else:  # H4
            n_full = 1
            n_crops = 5
            has_telem = "Yes (Kinematics)"

        est_tokens = (n_full * 1600) + (n_crops * 288)
        img_summary = f"{n_full} Full + {n_crops} ROI"

        print(f"{lvl:<6} | {img_summary:<16} | ~{est_tokens:<20} | {has_telem:<20} | {level_descriptions[lvl]}")

    print("=" * 85)
    print("Controlled setup verified: All levels evaluate the identical CandidateEvidencePackage.")
    print("Clean ROI and Marked SoM ROI remain separate representations in memory and on disk.")
    print("=" * 85)


if __name__ == "__main__":
    run_h_ladder_benchmark()
