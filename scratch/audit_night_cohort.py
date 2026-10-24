import os
import sys
import time
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_ROOT = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive"
CSV_PATH = os.path.join(ARCHIVE_ROOT, "metadata-real.csv")

def sample_video_luminance(rel_path: str):
    full_path = os.path.join(ARCHIVE_ROOT, rel_path)
    if not os.path.isfile(full_path):
        return {"path": rel_path, "status": "FILE_NOT_FOUND", "lums": [], "mean_lum": None}
    
    cap = cv2.VideoCapture(full_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if n_frames <= 0:
        cap.release()
        return {"path": rel_path, "status": "EMPTY_OR_UNREADABLE", "lums": [], "mean_lum": None}
    
    # 3 frames sampled at 25%, 50%, 75%
    sample_indices = [max(0, n_frames // 4), max(0, n_frames // 2), max(0, 3 * n_frames // 4)]
    lums = []
    for f in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lums.append(float(np.mean(gray)))
    cap.release()
    
    mean_lum = float(np.mean(lums)) if lums else None
    return {
        "path": rel_path,
        "n_frames": n_frames,
        "fps": fps,
        "sample_indices": sample_indices,
        "lums": lums,
        "mean_lum": mean_lum,
        "status": "OK"
    }

def main():
    print("Loading metadata-real.csv...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    day_counts = df['day_time'].value_counts().to_dict()
    print(f"Total rows: {total_rows}")
    print(f"day_time distribution: {day_counts}")
    
    night_df = df[df['day_time'] == 'night']
    day_df = df[df['day_time'] == 'day']
    
    night_count = len(night_df)
    print(f"Night count: {night_count} (cross-check against 651: {'MATCH' if night_count == 651 else 'MISMATCH'})")
    
    # Distributions
    region_dist = night_df['region'].value_counts().to_dict()
    scene_dist = night_df['scene_layout'].value_counts().to_dict()
    weather_dist = night_df['weather'].value_counts().to_dict()
    type_dist = night_df['type'].value_counts().to_dict()
    
    print("\n--- Region Distribution (Night) ---")
    for k, v in list(region_dist.items())[:10]:
        print(f"  {k}: {v}")
        
    print("\n--- Scene Layout Distribution (Night) ---")
    for k, v in scene_dist.items():
        print(f"  {k}: {v}")
        
    # Run luminance checks
    print(f"\nSampling luminance for all {night_count} night videos...")
    t0 = time.time()
    night_paths = night_df['path'].tolist()
    
    night_results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(sample_video_luminance, night_paths):
            night_results.append(res)
    t1 = time.time()
    print(f"Sampled {len(night_results)} night videos in {t1-t0:.2f}s")
    
    # Flag night videos with mean luminance > 100
    flagged_night_high = [r for r in night_results if r['mean_lum'] is not None and r['mean_lum'] > 100.0]
    print(f"Flagged night videos with mean luminance > 100: {len(flagged_night_high)}")
    for r in flagged_night_high:
        print(f"  POTENTIAL MISLABEL (Night but Lum > 100): {r['path']} -> mean_lum={r['mean_lum']:.2f}")

    # Now sample day videos to check if any have mean luminance < 60
    print(f"\nSampling luminance for all {len(day_df)} day videos...")
    t2 = time.time()
    day_paths = day_df['path'].tolist()
    day_results = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for res in ex.map(sample_video_luminance, day_paths):
            day_results.append(res)
    t3 = time.time()
    print(f"Sampled {len(day_results)} day videos in {t3-t2:.2f}s")
    
    flagged_day_low = [r for r in day_results if r['mean_lum'] is not None and r['mean_lum'] < 60.0]
    print(f"Flagged day videos with mean luminance < 60: {len(flagged_day_low)}")
    for r in flagged_day_low[:15]:
        print(f"  POTENTIAL MISLABEL (Day but Lum < 60): {r['path']} -> mean_lum={r['mean_lum']:.2f}")

    # Summary
    summary_data = {
        "total_rows": total_rows,
        "night_count": night_count,
        "preflight_cross_check_651": (night_count == 651),
        "day_counts": day_counts,
        "region_distribution_night": region_dist,
        "scene_layout_distribution_night": scene_dist,
        "weather_distribution_night": weather_dist,
        "type_distribution_night": type_dist,
        "flagged_night_high_lum_gt_100": flagged_night_high,
        "flagged_day_low_lum_lt_60": flagged_day_low,
    }
    
    out_json = "scratch/night_cohort_audit.json"
    with open(out_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved complete audit to {out_json}")

if __name__ == "__main__":
    main()
