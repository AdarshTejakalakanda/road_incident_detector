import os
import json
import glob
from collections import defaultdict, Counter
import numpy as np

ev_dir = r"evidence/2026-10-05/5uh8-3q51uI_00"

ep_dirs = sorted([os.path.join(ev_dir, d) for d in os.listdir(ev_dir) if os.path.isdir(os.path.join(ev_dir, d))])
print(f"Total episode directories: {len(ep_dirs)}")

target_frames = [42, 180, 495, 620]

print("\n--- EPISODES AT TARGET FRAMES ---")
for d in ep_dirs:
    meta_p = os.path.join(d, "metadata.json")
    with open(meta_p, "r", encoding="utf-8") as f:
        m = json.load(f)
    pf = m.get("peak_impact_frame")
    sf = m.get("start_frame")
    lf = m.get("last_trigger_frame")
    if any(tf in [pf, sf, lf] for tf in target_frames):
        print(f"Ep: {m.get('episode_id'):<18} | det: {m.get('primary_detector'):<25} | peak_f: {pf} | start_f: {sf} | last_f: {lf} | tracks: {m.get('involved_track_ids')}")

# Now let's collect ALL raw triggers and examine their acceleration, event_type, tracks, etc.
