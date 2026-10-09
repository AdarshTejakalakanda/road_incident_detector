import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)

v = [x for x in in_dist if x['video_id'] == 'iA_u73BG4ZM_00'][0]

print("Track histories in iA_u73BG4ZM_00:")
t_frames = {}
for f_idx, pts, tracks, kin, conv in v['frames']:
    for tid, tr in tracks.items():
        if tid not in t_frames:
            t_frames[tid] = []
        k = kin.get(tid)
        t_frames[tid].append((f_idx, tr.current_bbox_xywh, k.speed_kmh if k else 0, k.heading_deg if k else 0, k.acceleration_mag if k else 0))

for tid in sorted(t_frames.keys()):
    frames = [x[0] for x in t_frames[tid]]
    if any(90 <= f <= 135 for f in frames):
        print(f"\n--- Track {tid} (frames {min(frames)} to {max(frames)}, len={len(frames)}) ---")
        for f, bbox, spd, hd, acc in t_frames[tid]:
            if 100 <= f <= 125:
                print(f"  f={f:3d} | box={[round(x,1) for x in bbox]} | spd={spd:5.1f} | hd={hd:5.1f} | a={acc:6.1f}")
