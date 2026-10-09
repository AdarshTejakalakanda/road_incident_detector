import os
import json
import glob
from collections import defaultdict, Counter
import numpy as np

ev_dir = r"evidence/2026-10-05/5uh8-3q51uI_00"
ep_dirs = sorted([os.path.join(ev_dir, d) for d in os.listdir(ev_dir) if os.path.isdir(os.path.join(ev_dir, d))])

print(f"Total episode dirs: {len(ep_dirs)}")

episodes = []
detector_counts = Counter()
raw_triggers_total = 0
all_tracks = set()

for d in ep_dirs:
    meta_p = os.path.join(d, "metadata.json")
    if not os.path.exists(meta_p):
        continue
    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    det = meta.get("primary_detector")
    detector_counts[det] += 1
    raw_triggers_total += meta.get("raw_triggers_count", 1)
    tracks = meta.get("involved_track_ids", [])
    all_tracks.update(tracks)
    
    # Read telemetry
    tel_files = glob.glob(os.path.join(d, "telemetry", "*.json"))
    tel_data = []
    if tel_files:
        with open(tel_files[0], "r", encoding="utf-8") as f:
            tel_data = json.load(f)
            
    episodes.append({
        "dir": os.path.basename(d),
        "meta": meta,
        "tel": tel_data,
    })

print(f"Total raw triggers in metadata: {raw_triggers_total}")
print(f"Detector breakdown:")
for d, c in detector_counts.most_common():
    print(f"  {d:<30}: {c}")

print(f"Max track ID involved: {max(all_tracks) if all_tracks else None}")
