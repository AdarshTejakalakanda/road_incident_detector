import os
import sys
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_ROOT = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive"
CSV_PATH = os.path.join(ARCHIVE_ROOT, "metadata-real.csv")

def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()

def compute_clip_record(args):
    idx, row = args
    rel_path = str(row['path'])
    full_path = os.path.join(ARCHIVE_ROOT, rel_path)
    clip_id = os.path.basename(rel_path).replace(".mp4", "")
    
    # 1. SHA-256
    file_hash = sha256_file(full_path)
    
    # 2. Luminance (3 evenly spaced frames at 25%, 50%, 75%)
    cap = cv2.VideoCapture(full_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    
    day_time = str(row['day_time']).strip().lower()
    split_in_dist = str(row.get('split_in_distribution', 'MISSING'))
    split_geo = str(row.get('split_geo_aware', 'MISSING'))
    region = str(row.get('region', 'MISSING'))
    
    # camera ID and date: check if camera_id or date is in metadata-real.csv
    # 'region' is the closest camera/region tag, date is not in metadata-real.csv
    camera_id = region if region != 'nan' else 'MISSING'
    date = 'MISSING'
    
    # Cohort categorization
    if day_time == 'night':
        if mean_lum is not None and mean_lum <= 100.0:
            cohort = "verified-night"
        else:
            cohort = "mislabeled-night"
    elif day_time == 'day':
        if mean_lum is not None and mean_lum < 60.0:
            cohort = "excluded"
        else:
            cohort = "day"
    else:
        cohort = "excluded"
        
    return {
        "clip_id": clip_id,
        "metadata_row_idx": int(idx),
        "path": rel_path,
        "day_time": day_time,
        "measured_mean_luminance": round(mean_lum, 3) if mean_lum is not None else None,
        "luminance_method": "mean_y_over_3_frames_at_25_50_75_percent",
        "sha256": file_hash,
        "split_in_distribution": split_in_dist,
        "split_geo_aware": split_geo,
        "camera_id": camera_id,
        "date": date,
        "cohort": cohort,
        "scene_layout": str(row.get('scene_layout', 'MISSING')),
        "accident_frame": int(row.get('accident_frame', -1)),
        "accident_time": float(row.get('accident_time', -1.0)),
        "weather": str(row.get('weather', 'MISSING')),
    }

def main():
    print("Building full cohort manifests...")
    df = pd.read_csv(CSV_PATH)
    t0 = time.time()
    
    tasks = [(idx, row) for idx, row in df.iterrows()]
    records = []
    
    with ThreadPoolExecutor(max_workers=16) as ex:
        for rec in ex.map(compute_clip_record, tasks):
            records.append(rec)
            if len(records) % 200 == 0:
                print(f"  Processed {len(records)} / {len(tasks)} clips...")
                
    t1 = time.time()
    print(f"Processed all {len(records)} clips in {t1-t0:.2f}s")
    
    cohorts = {
        "verified-night": [r for r in records if r["cohort"] == "verified-night"],
        "mislabeled-night": [r for r in records if r["cohort"] == "mislabeled-night"],
        "day": [r for r in records if r["cohort"] == "day"],
        "excluded": [r for r in records if r["cohort"] == "excluded"],
    }
    
    for cname, clist in cohorts.items():
        print(f"Cohort '{cname}': {len(clist)} clips")
        
    # Check duplicate hashes
    all_hashes = [r["sha256"] for r in records]
    dup_hashes = [h for h, count in pd.Series(all_hashes).value_counts().items() if count > 1]
    print(f"Duplicate file hashes: {len(dup_hashes)}")
    
    # Save manifests
    manifest_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_clips": len(records),
        "cohort_counts": {k: len(v) for k, v in cohorts.items()},
        "duplicate_hashes_count": len(dup_hashes),
        "cohorts": cohorts
    }
    
    manifest_bytes = json.dumps(manifest_payload, indent=2).encode('utf-8')
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_payload["manifest_sha256"] = manifest_hash
    
    out_path = "scratch/cohort_manifests.json"
    with open(out_path, "wb") as f:
        f.write(json.dumps(manifest_payload, indent=2).encode('utf-8'))
        
    print(f"Manifest written to {out_path} with manifest SHA-256: {manifest_hash}")

if __name__ == "__main__":
    main()
