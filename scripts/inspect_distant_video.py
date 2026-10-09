import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)

v = [x for x in in_dist if x['video_id'] == 'iA_u73BG4ZM_00'][0]
print(f"Video: {v['video_id']}, GT Frame: {v['gt_frame']}, FPS: {v['fps']:.2f}, Total frames: {len(v['frames'])}")

for f_idx, pts, tracks, kin, conv in v['frames']:
    if 95 <= f_idx <= 135:
        t_info = []
        for tid, tr in tracks.items():
            k = kin.get(tid)
            if k:
                t_info.append(f"id={tid}({tr.class_label}, box={[round(x,1) for x in tr.current_bbox_xywh]}, spd={k.speed_kmh:.1f}, hd={k.heading_deg:.1f}, a={k.acceleration_mag:.1f})")
        conv_info = []
        for c in conv:
            conv_info.append(f"({c.track_id_1}<->{c.track_id_2}: dist={c.distance_m:.1f}, conv={c.is_converging}, tca={c.tca_sec:.2f}s, dca={c.dca_m:.1f})")
        print(f"\nf={f_idx}:")
        print("  Tracks:", "; ".join(t_info))
        if conv_info:
            print("  Convergences:", "; ".join(conv_info))
