import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle

with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    vids = pickle.load(f)

v = [x for x in vids if x["video_id"] == "LiM5LPSZoow_00"][0]

for f_idx, pts, tr, kin, conv in v["frames"]:
    if f_idx in [15, 16, 17]:
        for c in conv:
            if (c.track_id_1, c.track_id_2) in [(12, 3), (3, 12), (5, 13), (13, 5)]:
                t1, t2 = tr.get(c.track_id_1), tr.get(c.track_id_2)
                k1, k2 = kin.get(c.track_id_1), kin.get(c.track_id_2)
                print(f"f={f_idx} pair=({c.track_id_1},{c.track_id_2}) dist_m={c.distance_m:.1f} b1={t1.current_bbox_xywh if t1 else None} b2={t2.current_bbox_xywh if t2 else None} a1={k1.acceleration_mag if k1 else 0} a2={k2.acceleration_mag if k2 else 0}")
