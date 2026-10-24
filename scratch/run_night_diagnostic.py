"""Detector Night-Footage Diagnostic Pass.

Read-only diagnostic execution for Arms A-E across raw and enhanced frames
at conf thresholds 0.10 and 0.25.
"""

import os
import sys
import json
import hashlib
import time
from typing import Dict, List, Any
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.run_annotated_detector import get_adapter
from road_surveillance.preprocessing.illumination import LowLightPreprocessor, IlluminationEstimator

# 1. Verification of models & hashes
EXPECTED_HASHES = {
    "Arm A": ("models/yolo26n.pt", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "Arm B": ("models/benchmark_arms/UVH-26-MV-YOLOv11-S.pt", "12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150"),
    "Arm C": ("models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth", "da733d3e3934d5d0c10766ae06f33d0d8e6845902decf507b9ef22e02c854bb1"),
    "Arm D": ("models/benchmark_arms/vietnguyen_best.pt", "486f5c4c4d3bf0d01134b591c42182517a9ca7cf1ab91799c877bfe723a0d747"),
    "Arm E": ("models/benchmark_arms/night-vehicle-yolov8s.pt", "3e7d5cfa87d3d7a6ad8da8319bf36070d661e3643e927f7a32476471a8d82b48"),
}

def sha256_file(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_hashes():
    results = {}
    for arm, (path, expected) in EXPECTED_HASHES.items():
        actual = sha256_file(path)
        match = (actual == expected)
        results[arm] = {"path": path, "expected": expected, "actual": actual, "match": match}
    return results

def run_diagnostic():
    hash_check = verify_hashes()
    for arm, info in hash_check.items():
        assert info["match"], f"Hash mismatch for {arm}: expected {info['expected']}, got {info['actual']}"
    print("Task 0 Hash verification PASSED.")

    # Load adapters
    print("Loading all 5 adapters...")
    arms = ["A", "B", "C", "D", "E"]
    adapters = {arm: get_adapter(arm) for arm in arms}

    # Low light preprocessor (production setting: N3 with threshold 50.0, gamma 0.65)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)

    # Videos to analyze:
    # Set 1: Recommended clip 5uh8-3q51uI_00 (day control frame 50, frames 500, 520, 540, 560, 580, 600)
    # Set 2: Genuine nocturnal clip Z4kg2Ev3vhk_00 (frames 50, 100, 150, 170, 200, 250)
    clip_5uh8_path = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\5uh8-3q51uI_00.mp4"
    clip_z4kg_path = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4"

    experiments = [
        {
            "suite_name": "5uh8-3q51uI_00 (User-Recommended)",
            "video_path": clip_5uh8_path,
            "day_control_frame": 50,
            "night_frames": [500, 520, 540, 560, 580, 600],
        },
        {
            "suite_name": "Z4kg2Ev3vhk_00 (Metadata-Tagged Nocturnal)",
            "video_path": clip_z4kg_path,
            "day_control_frame": 50, # We will use 5uh8 frame 50 as daytime control
            "day_control_video": clip_5uh8_path,
            "night_frames": [50, 100, 150, 170, 200, 250],
        }
    ]

    all_results = {}

    for exp in experiments:
        suite = exp["suite_name"]
        print(f"\n==========================================")
        print(f"Running Diagnostic Suite: {suite}")
        print(f"==========================================")

        cap_night = cv2.VideoCapture(exp["video_path"])
        if "day_control_video" in exp:
            cap_day = cv2.VideoCapture(exp["day_control_video"])
        else:
            cap_day = cap_night

        # 1. Extract daytime control frame
        cap_day.set(cv2.CAP_PROP_POS_FRAMES, exp["day_control_frame"])
        ret, day_raw = cap_day.read()
        assert ret, f"Failed to read day control frame {exp['day_control_frame']}"
        day_raw_lum = IlluminationEstimator.estimate_mean_luminance(day_raw)
        day_enh, day_telem = preprocessor.process(day_raw)
        day_enh_lum = IlluminationEstimator.estimate_mean_luminance(day_enh)

        frame_data = {
            "day_control": {
                "frame_idx": exp["day_control_frame"],
                "raw_lum": day_raw_lum,
                "enh_lum": day_enh_lum,
                "is_low_light": day_telem["applied"],
                "raw_img": day_raw,
                "enh_img": day_enh,
            },
            "night_frames": []
        }

        for f_idx in exp["night_frames"]:
            cap_night.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, n_raw = cap_night.read()
            assert ret, f"Failed to read night frame {f_idx}"
            n_raw_lum = IlluminationEstimator.estimate_mean_luminance(n_raw)
            n_enh, n_telem = preprocessor.process(n_raw)
            n_enh_lum = IlluminationEstimator.estimate_mean_luminance(n_enh)
            frame_data["night_frames"].append({
                "frame_idx": f_idx,
                "raw_lum": n_raw_lum,
                "enh_lum": n_enh_lum,
                "is_low_light": n_telem["applied"],
                "raw_img": n_raw,
                "enh_img": n_enh,
            })

        # Run detection grid
        # For each arm x each frame x each threshold (0.10, 0.25) x (raw / enhanced)
        grid_results = {}
        for arm_id in arms:
            adapter = adapters[arm_id]
            print(f"Running arm {arm_id} ({adapter.model_name})...")
            grid_results[arm_id] = {
                "day": {},
                "night": []
            }

            # Day control frame
            d_entry = frame_data["day_control"]
            for cond, img in [("raw", d_entry["raw_img"]), ("enh", d_entry["enh_img"])]:
                for tau in [0.10, 0.25]:
                    out = adapter.detect(img, conf_threshold=tau, frame_id=d_entry["frame_idx"])
                    scores = out.scores.tolist() if len(out.scores) > 0 else []
                    grid_results[arm_id]["day"][f"{cond}_{tau}"] = {
                        "count": len(out.boxes),
                        "scores": scores,
                        "classes": out.class_names,
                        "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                        "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0,
                        "min_score": float(np.min(scores)) if len(scores) > 0 else 0.0,
                    }

            # Night frames
            for n_entry in frame_data["night_frames"]:
                n_res = {"frame_idx": n_entry["frame_idx"]}
                for cond, img in [("raw", n_entry["raw_img"]), ("enh", n_entry["enh_img"])]:
                    for tau in [0.10, 0.25]:
                        out = adapter.detect(img, conf_threshold=tau, frame_id=n_entry["frame_idx"])
                        scores = out.scores.tolist() if len(out.scores) > 0 else []
                        n_res[f"{cond}_{tau}"] = {
                            "count": len(out.boxes),
                            "scores": scores,
                            "classes": out.class_names,
                            "mean_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                            "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0,
                            "min_score": float(np.min(scores)) if len(scores) > 0 else 0.0,
                        }
                grid_results[arm_id]["night"].append(n_res)

        # Remove image buffers before persisting
        del frame_data["day_control"]["raw_img"]
        del frame_data["day_control"]["enh_img"]
        for nf in frame_data["night_frames"]:
            del nf["raw_img"]
            del nf["enh_img"]

        all_results[suite] = {
            "frame_data": frame_data,
            "grid_results": grid_results,
        }

    # Save to json
    out_path = "scratch/night_diagnostic_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDiagnostic completed. Results saved to {out_path}")

if __name__ == "__main__":
    run_diagnostic()
