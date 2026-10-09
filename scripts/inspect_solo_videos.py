import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import numpy as np

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)
with open('cache_videos_iter_b_split_geo_aware_15.pkl', 'rb') as f:
    geo = pickle.load(f)

solo_vids = {
    'nAXTthLfgtI_00': (62, 'in_dist'),
    'vxZ47DoPNY8_00': (50, 'in_dist'),
    'VozrseCzhQc_00': (84, 'in_dist'),
    'UarP8qU1S-c_00': (130, 'geo'),
    'UarP8qU1S-c_01': (30, 'geo'),
}

all_vids = {v['video_id']: v for v in in_dist}
all_vids.update({v['video_id']: v for v in geo})

for vid_id, (gt_f, split) in solo_vids.items():
    v = all_vids[vid_id]
    print(f"\n==================================================")
    print(f"VIDEO: {vid_id} (GT Frame: {gt_f}, FPS: {v['fps']:.1f})")
    print(f"==================================================")
    
    # Track historical data
    track_hist = {} # tid -> list of (f_idx, bbox, speed, accel, heading)
    
    for f_idx, pts, tracks, kin, conv in v['frames']:
        for tid, tr in tracks.items():
            k = kin.get(tid)
            if not k:
                continue
            bx, by, bw, bh = tr.current_bbox_xywh
            aspect = bw / max(1.0, bh)
            if tid not in track_hist:
                track_hist[tid] = []
            track_hist[tid].append({
                'f': f_idx,
                'bbox': (bx, by, bw, bh),
                'aspect': aspect,
                'speed': k.speed_kmh,
                'accel': k.acceleration_mag,
                'heading': k.heading_deg,
                'label': tr.class_label,
                'age': tr.track_age
            })

    # Analyze tracks that were active around gt_frame
    for tid, hist in track_hist.items():
        frames_present = [h['f'] for h in hist]
        if not (min(frames_present) <= gt_f + 35 and max(frames_present) >= gt_f - 35):
            continue
        
        # Look at window around gt_frame
        window = [h for h in hist if gt_f - 35 <= h['f'] <= gt_f + 35]
        if not window:
            continue
        
        print(f"\n--- Track ID {tid} ({window[0]['label']}), active frames {min(frames_present)} to {max(frames_present)} ---")
        
        # Calculate max decel, max heading change, aspect ratio change
        speeds = [h['speed'] for h in window]
        accels = [h['accel'] for h in window]
        aspects = [h['aspect'] for h in window]
        headings = [h['heading'] for h in window]
        
        # Heading delta between consecutive frames
        heading_deltas = []
        for i in range(1, len(window)):
            dh = abs(window[i]['heading'] - window[i-1]['heading'])
            if dh > 180:
                dh = 360 - dh
            heading_deltas.append(dh)
            
        print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h")
        print(f"  Min acceleration (max decel): {min(accels):.1f} m/s^2")
        print(f"  Aspect ratio range: {min(aspects):.2f} - {max(aspects):.2f}")
        print(f"  Max 1-frame heading change: {max(heading_deltas) if heading_deltas else 0:.1f} deg")
        
        # Print frame by frame within +/- 15 frames of gt_f
        print("  Key frames near GT:")
        for h in window:
            if abs(h['f'] - gt_f) <= 15:
                print(f"    f={h['f']:3d}: bbox=({h['bbox'][0]:.0f},{h['bbox'][1]:.0f},{h['bbox'][2]:.0f},{h['bbox'][3]:.0f}) ar={h['aspect']:.2f} spd={h['speed']:5.1f} km/h acc={h['accel']:6.1f} m/s^2 head={h['heading']:5.1f} deg age={h['age']}")
