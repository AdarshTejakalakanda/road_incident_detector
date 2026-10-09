import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle
import numpy as np

with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    vids = pickle.load(f)
v = [x for x in vids if x["video_id"] == "kABWTDKgc9U_00"][0]

print("Checking all vehicle pairs in kABWTDKgc9U_00 between frame 180 and 215:")
min_dist_pairs = []
for f_idx, pts, tr, kin, conv in v["frames"]:
    if 180 <= f_idx <= 215:
        # compute all pairwise pixel distances
        tids = list(tr.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                id1, id2 = tids[i], tids[j]
                t1, t2 = tr[id1], tr[id2]
                b1, b2 = t1.current_bbox_xywh, t2.current_bbox_xywh
                c1 = (b1[0] + b1[2]/2, b1[1] + b1[3]/2)
                c2 = (b2[0] + b2[2]/2, b2[1] + b2[3]/2)
                d_px = np.hypot(c1[0]-c2[0], c1[1]-c2[1])
                char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3]))/2)
                if d_px <= 2.5 * char_dim:
                    min_dist_pairs.append((f_idx, id1, id2, d_px, char_dim, t1.class_label, t2.class_label, t1.track_age, t2.track_age))

print(f"Close proximity vehicle pairs found: {len(min_dist_pairs)}")
for p in min_dist_pairs[:20]:
    print(f"f={p[0]} pair=({p[1]},{p[2]}) dist_px={p[3]:.1f} char={p[4]:.1f} cls=({p[5]},{p[6]}) age=({p[7]},{p[8]})")
