import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

with open('cache_videos_iter_b_split_geo_aware_15.pkl', 'rb') as f:
    geo = pickle.load(f)

vids = {v['video_id']: v for v in geo}
v0 = vids['UarP8qU1S-c_00']
v1 = vids['UarP8qU1S-c_01']

print("=== Tracing UarP8qU1S-c_00 (GT=130) ===")
for tid in [1, 2]:
    for f_idx, pts, tracks, kin, conv in v0['frames']:
        if tid in tracks:
            tr = tracks[tid]
            k = kin.get(tid)
            if k:
                w, h = tr.current_bbox_xywh[2], tr.current_bbox_xywh[3]
                scale = 4.5 / max(15.0, max(w, h))
                ma = k.acceleration_mag * scale
                mspd = k.speed_kmh * scale
                if 95 <= f_idx <= 135:
                    print(f"tid={tid} f={f_idx:3d} tr_age={tr.track_age} hist_len={len(tr.ground_history)} ma={ma:6.2f} mspd={mspd:5.2f} hd={k.heading_deg:5.1f}")

print("\n=== Tracing UarP8qU1S-c_01 (GT=30) ===")
for tid in [1, 2]:
    for f_idx, pts, tracks, kin, conv in v1['frames']:
        if tid in tracks:
            tr = tracks[tid]
            k = kin.get(tid)
            if k:
                w, h = tr.current_bbox_xywh[2], tr.current_bbox_xywh[3]
                scale = 4.5 / max(15.0, max(w, h))
                ma = k.acceleration_mag * scale
                mspd = k.speed_kmh * scale
                if 30 <= f_idx <= 70:
                    print(f"tid={tid} f={f_idx:3d} tr_age={tr.track_age} hist_len={len(tr.ground_history)} ma={ma:6.2f} mspd={mspd:5.2f} hd={k.heading_deg:5.1f}")
