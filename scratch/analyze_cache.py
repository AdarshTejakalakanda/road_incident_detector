import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pickle

for split in ["in_distribution", "geo_aware"]:
    print(f"\n--- Split: {split} ---")
    with open(f"cache_videos_iter_b_split_{split}_15.pkl", "rb") as f:
        vids = pickle.load(f)
    for i, v in enumerate(vids):
        px_speeds = [k.pixel_speed for f in v['frames'] for k in f[3].values()]
        px_accels = [k.pixel_acceleration for f in v['frames'] for k in f[3].values()]
        print(f"{i}: {v['video_id']} - k_count={len(px_speeds)}, max_px_spd={max(px_speeds) if px_speeds else 0:.1f}, max_px_acc={max(px_accels) if px_accels else 0:.1f}")
