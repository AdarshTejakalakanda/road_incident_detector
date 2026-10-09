import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)

v = [x for x in in_dist if x['video_id'] == 'iA_u73BG4ZM_00'][0]
print(f"=== iA_u73BG4ZM_00 (GT={v['gt_frame']}) ===")

# Print all frames where ANY track exists between 70 and 150
for f_idx, pts, tracks, kin, conv in v['frames']:
    if 70 <= f_idx <= 150 and tracks:
        t_summary = []
        for tid, tr in tracks.items():
            k = kin.get(tid)
            spd = k.speed_kmh if k else 0
            hd = k.heading_deg if k else 0
            acc = k.acceleration_mag if k else 0
            t_summary.append(f"id={tid}({tr.class_label}, age={tr.track_age}, box={[round(x,1) for x in tr.current_bbox_xywh]}, spd={spd:.1f}, hd={hd:.1f}, a={acc:.1f})")
        print(f"f={f_idx:3d}: {'; '.join(t_summary)}")
