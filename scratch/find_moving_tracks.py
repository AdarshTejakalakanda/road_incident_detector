import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle

with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    vids = pickle.load(f)
v = [x for x in vids if x["video_id"] == "kABWTDKgc9U_00"][0]

for f_idx, pts, tr, kin, conv in v["frames"]:
    if 190 <= f_idx <= 205:
        for tid, k in kin.items():
            if abs(k.acceleration_mag) > 5.0 or k.speed_kmh > 15.0:
                t = tr[tid]
                print(f"f={f_idx} tid={tid:2d} ({t.class_label:10s}): spd={k.speed_kmh:6.1f} acc={k.acceleration_mag:6.1f} box={t.current_bbox_xywh}")
