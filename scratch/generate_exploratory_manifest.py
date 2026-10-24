import os
import sys
import json
import random
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_ROOT = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive"
CSV_PATH = os.path.join(ARCHIVE_ROOT, "metadata-real.csv")

def main():
    with open("scratch/clean_cohort_manifests.json") as f:
        clean_manifest = json.load(f)
        
    df = pd.read_csv(CSV_PATH)
    
    random.seed(42)
    
    # Candidate pools from both_train only (strictly avoiding test splits and quarantine)
    vn_pool = [x for x in clean_manifest['cohorts']['verified-night'] if x['split_in_distribution'] == 'train' and x['split_geo_aware'] == 'train']
    mn_pool = [x for x in clean_manifest['cohorts']['mislabeled-night'] if x['split_in_distribution'] == 'train' and x['split_geo_aware'] == 'train']
    day_pool = [x for x in clean_manifest['cohorts']['day'] if x['split_in_distribution'] == 'train' and x['split_geo_aware'] == 'train']
    
    vn_pool.sort(key=lambda x: x['clip_id'])
    mn_pool.sort(key=lambda x: x['clip_id'])
    day_pool.sort(key=lambda x: x['clip_id'])
    
    vn_sample = random.sample(vn_pool, 12)
    mn_sample = random.sample(mn_pool, 6)
    day_sample = random.sample(day_pool, 12)
    
    selected_records = []
    for item in vn_sample + mn_sample + day_sample:
        row = df.loc[item['metadata_row_idx']]
        selected_records.append({
            'clip_id': item['clip_id'],
            'path': item['path'],
            'cohort': item['cohort'],
            'sha256': item['sha256'],
            'measured_mean_luminance': item['measured_mean_luminance'],
            'duration_s': float(row['duration']),
            'no_frames': int(row['no_frames']),
            'accident_frame': int(row['accident_frame']),
            'accident_time': float(row['accident_time']),
            'incident_type': str(row['type']),
            'weather': str(row['weather']),
            'scene_layout': str(row['scene_layout']),
            'region': str(row['region']),
            'split_in_distribution': item['split_in_distribution'],
            'split_geo_aware': item['split_geo_aware'],
        })
        
    manifest_payload = {
        "experiment_name": "30-video exploratory detector-parameter optimization experiment",
        "selection_seed": 42,
        "selection_rationale": "Stratified random sample of 30 development videos (12 verified-night, 6 mislabeled-night, 12 day) from pure development splits (both split_in_distribution == train AND split_geo_aware == train). Zero overlap with any test partition; all 27 historical tuning/calibration clips strictly excluded.",
        "sample_size": len(selected_records),
        "cohort_counts": {
            "verified-night": 12,
            "mislabeled-night": 6,
            "day": 12
        },
        "total_frames": sum(x['no_frames'] for x in selected_records),
        "total_duration_s": round(sum(x['duration_s'] for x in selected_records), 2),
        "videos": selected_records
    }
    
    out_path = "scratch/exploratory_30_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)
        
    print(f"Sample manifest successfully created at {out_path} with {len(selected_records)} videos.")

if __name__ == "__main__":
    main()
