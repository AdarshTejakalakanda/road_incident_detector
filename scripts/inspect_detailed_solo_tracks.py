import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)
with open('cache_videos_iter_b_split_geo_aware_15.pkl', 'rb') as f:
    geo = pickle.load(f)

vids = {v['video_id']: v for v in in_dist}
vids.update({v['video_id']: v for v in geo})

for vid in ['vxZ47DoPNY8_00', 'UarP8qU1S-c_00', 'UarP8qU1S-c_01']:
    v = vids[vid]
    print(f"\n=======================================================")
    print(f"VIDEO: {vid} (GT={v['gt_frame']}, FPS={v['fps']:.2f})")
    print(f"=======================================================")
    
    # Check all tracks
    all_tids = set()
    for f_idx, pts, tracks, kin, conv in v['frames']:
        for tid, tr in tracks.items():
            all_tids.add(tid)
            
    for tid in sorted(all_tids):
        hist = []
        for f_idx, pts, tracks, kin, conv in v['frames']:
            if tid in tracks:
                tr = tracks[tid]
                k = kin.get(tid)
                bx, by, bw, bh = tr.current_bbox_xywh
                hist.append((f_idx, tr.class_label, (bx, by, bw, bh), bw / max(1.0, bh), k.speed_kmh if k else 0, k.acceleration_mag if k else 0, k.heading_deg if k else 0))
        
        frames = [h[0] for h in hist]
        if any(abs(f - v['gt_frame']) <= 35 for f in frames):
            print(f"\nTrack {tid} ({hist[0][1]}), frames {min(frames)} to {max(frames)} (total {len(frames)} frames):")
            for f, lbl, bbox, ar, spd, acc, hd in hist:
                if abs(f - v['gt_frame']) <= 35:
                    print(f"  f={f:3d} | bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] | AR={ar:.2f} | spd={spd:5.1f} km/h | acc={acc:6.1f} m/s^2 | hd={hd:5.1f}°")
