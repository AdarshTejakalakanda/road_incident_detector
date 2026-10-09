import os
import json
import glob
from collections import defaultdict, Counter
import numpy as np

ev_dir = r"evidence/2026-10-05/5uh8-3q51uI_00"
target_frames = [42, 180, 495, 620]

telemetry_files = glob.glob(os.path.join(ev_dir, "*", "telemetry", "*.json"))
print(f"Total telemetry files found: {len(telemetry_files)}")

# 1. Search for target frames
target_data = defaultdict(dict)
for tf in telemetry_files:
    try:
        with open(tf, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                f_idx = item.get("frame_idx")
                if f_idx in target_frames:
                    for tid, tinfo in item.get("tracks", {}).items():
                        target_data[f_idx][tid] = tinfo
    except Exception as e:
        pass

for f_idx in sorted(target_frames):
    print(f"\n=== FRAME {f_idx} TRACKS ({len(target_data[f_idx])} tracks) ===")
    for tid, tinfo in sorted(target_data[f_idx].items(), key=lambda x: int(x[0])):
        acc = tinfo.get("accel_mag", 0.0)
        spd = tinfo.get("speed_kmh", 0.0)
        bbox = [round(x, 1) for x in tinfo.get("bbox_xywh", [])]
        gpos = [round(x, 2) for x in tinfo.get("ground_pos", [])]
        print(f"  Track {tid:>3}: spd={spd:>6.1f} km/h, acc={acc:>7.1f} m/s^2, bbox={bbox}, ground={gpos}")
