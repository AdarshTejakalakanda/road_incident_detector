import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from scratch.classify_fps_and_tps import fp_records, tp_list, all_cached_videos
from scripts.benchmark_iteration_b_detector_recovery import compute_iou

print("=========================================================================================")
print("DETAILED ANALYSIS OF ALL 22 FP EPISODES")
print("=========================================================================================")

for i, r in enumerate(fp_records, 1):
    vid_id = r["video_id"]
    split = r["split"]
    v = all_cached_videos[(split, vid_id)]
    f_idx = r["trigger_frame"]
    tids = r["track_ids"]
    dets = r["detectors"]
    diff = r["diff_frames"]
    gt_f = r["gt_frame"]
    fps = v["fps"]
    
    fr_tuple = next((f for f in v["frames"] if f[0] == f_idx), None)
    if not fr_tuple:
        continue
    _, pts_ns, tracks, kin, conv = fr_tuple
    
    print(f"\n--- FP #{i:02d}: {vid_id} ({split}) @ f={f_idx} (GT={gt_f}, diff={diff:+d}f / {diff/fps:+.2f}s) | det={dets} ---")
    
    # Check each track involved
    for tid in tids:
        tr = tracks.get(tid)
        k = kin.get(tid)
        if not tr:
            print(f"  Track {tid}: NOT FOUND")
            continue
        
        # Pixel kinematics from history
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
                        
        w, h = tr.current_bbox_xywh[2], tr.current_bbox_xywh[3]
        char_dim = max(15.0, max(w, h))
        px_scale = 4.5 / char_dim
        v_px_scaled_kmh = px_spd * px_scale * 3.6
        a_px_scaled_ms2 = px_acc * px_scale
        
        gx, gy = k.ground_pos if k else (0, 0)
        # Horizon & coordinate validity check
        is_ground_valid = (abs(gx) <= 150.0 and -50.0 <= gy <= 250.0)
        
        gc_u, gc_v = tr.calculate_ground_contact(tr.current_bbox_xywh)
        
        print(f"  Track {tid} ({tr.class_label}): age={tr.track_age}, bbox=({tr.current_bbox_xywh[0]:.1f}, {tr.current_bbox_xywh[1]:.1f}, {w:.1f}, {h:.1f}), gc=({gc_u:.1f}, {gc_v:.1f})")
        print(f"    Ground: ({gx:.1f}, {gy:.1f}) [valid={is_ground_valid}]")
        print(f"    Pixel kin: spd={px_spd:.1f}px/s (scaled={v_px_scaled_kmh:.1f}km/h), acc={px_acc:.1f}px/s2 (scaled={a_px_scaled_ms2:.1f}m/s2)")
        if k:
            print(f"    Old cache kin: spd={k.speed_kmh:.1f}km/h, a_mag={k.acceleration_mag:.1f}m/s2")

    if len(tids) == 2:
        tr1, tr2 = tracks.get(tids[0]), tracks.get(tids[1])
        if tr1 and tr2:
            iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            c1 = (b1[0] + b1[2]/2, b1[1] + b1[3]/2)
            c2 = (b2[0] + b2[2]/2, b2[1] + b2[3]/2)
            px_dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
            char_avg = (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0
            norm_px_dist = px_dist / char_avg
            print(f"  Pair relation: IoU={iou:.3f}, PxCenterDist={px_dist:.1f}px, NormPxDist={norm_px_dist:.2f} body-lengths")
