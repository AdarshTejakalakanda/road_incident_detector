import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from scratch.classify_fps_and_tps import fp_records, tp_list, all_cached_videos

print("=== COMPUTING TRUE PIXEL KINEMATICS FOR 22 FP EPISODES ===")

for i, r in enumerate(fp_records, 1):
    vid_id = r["video_id"]
    split = r["split"]
    v = all_cached_videos[(split, vid_id)]
    f_idx = r["trigger_frame"]
    tids = r["track_ids"]
    dets = r["detectors"]
    
    print(f"\nFP #{i}: {vid_id} @ f={f_idx} (GT={r['gt_frame']}, diff={r['diff_frames']:+d}) dets={dets} tids={tids}")
    
    # Let's inspect the tracks
    fr_tuple = next((f for f in v["frames"] if f[0] == f_idx), None)
    if not fr_tuple:
        continue
    _, pts_ns, tracks, kin, conv = fr_tuple
    
    for tid in tids:
        tr = tracks.get(tid)
        k = kin.get(tid)
        if not tr:
            continue
        
        # Calculate pixel kinematics directly from tr.position_history and tr.timestamps
        # Find which index in tr.timestamps corresponds to pts_ns
        if pts_ns in tr.timestamps:
            idx = tr.timestamps.index(pts_ns)
            pos_hist = tr.position_history[:idx+1]
            time_hist = tr.timestamps[:idx+1]
        else:
            pos_hist = tr.position_history
            time_hist = tr.timestamps
        
        px_spd = 0.0
        px_acc = 0.0
        if len(pos_hist) >= 2 and len(time_hist) >= 2:
            dt = (time_hist[-1] - time_hist[-2]) * 1e-9
            if dt > 1e-5:
                dp = np.hypot(pos_hist[-1][0] - pos_hist[-2][0], pos_hist[-1][1] - pos_hist[-2][1])
                px_spd = dp / dt
                if len(pos_hist) >= 3 and len(time_hist) >= 3:
                    dt_prev = (time_hist[-2] - time_hist[-3]) * 1e-9
                    if dt_prev > 1e-5:
                        dp_prev = np.hypot(pos_hist[-2][0] - pos_hist[-3][0], pos_hist[-2][1] - pos_hist[-3][1])
                        px_spd_prev = dp_prev / dt_prev
                        px_acc = (px_spd - px_spd_prev) / dt
        
        # Also check ground validity with new rule:
        gx, gy = k.ground_pos if k else (0, 0)
        is_ground_valid = (abs(gx) <= 150.0 and -50.0 <= gy <= 250.0)
        
        w, h = tr.current_bbox_xywh[2], tr.current_bbox_xywh[3]
        char_dim = max(15.0, max(w, h))
        # Normalized pixel speed and acceleration
        norm_px_spd = px_spd / char_dim # body-lengths per sec
        norm_px_acc = px_acc / char_dim # body-lengths per sec^2
        
        print(f"  Track {tid} ({tr.class_label}): char_dim={char_dim:.1f}px")
        print(f"    Raw ground: ({gx:.1f}, {gy:.1f}) -> is_ground_valid: {is_ground_valid}")
        print(f"    Pixel spd: {px_spd:.1f} px/s ({norm_px_spd:.2f} body/s)")
        print(f"    Pixel acc: {px_acc:.1f} px/s^2 ({norm_px_acc:.2f} body/s^2)")
        if k:
            print(f"    Metric spd in cache: {k.speed_kmh:.1f} km/h | accel: {k.acceleration_mag:.1f} m/s^2")
