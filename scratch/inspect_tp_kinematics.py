import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from scratch.classify_fps_and_tps import tp_list, all_cached_videos

print("=== INSPECTING TRUE POSITIVES (12 TP EPISODES) ===")

for i, tp in enumerate(tp_list, 1):
    vid_id = tp["video_id"]
    split = tp["split"]
    v = all_cached_videos[(split, vid_id)]
    gt_f = tp["gt_frame"]
    cand_f = tp["cand_frame"]
    diff_f = tp["diff_frames"]
    primary_det = tp["primary_detector"]
    
    print(f"\nTP #{i}: {vid_id} ({split}) | GT={gt_f}, Cand={cand_f} (diff={diff_f:+d}f) | Primary={primary_det}")
    
    # Candidate matches
    tp_cands = [c for c in tp["candidates"] if abs(c["f"] - gt_f) <= 35]
    print(f"  Matched candidates count: {len(tp_cands)}")
    
    for c in tp_cands[:3]:
        cf = c["f"]
        tids = c["tids"]
        det = c["det"]
        
        fr_tuple = next((f for f in v["frames"] if f[0] == cf), None)
        if not fr_tuple:
            continue
        _, pts_ns, tracks, kin, conv = fr_tuple
        
        print(f"  -> Candidate @ f={cf} det={det} tids={tids}:")
        for tid in tids:
            tr = tracks.get(tid)
            k = kin.get(tid)
            if not tr:
                continue
            
            # Pixel kinematics
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
            norm_px_spd = px_spd / char_dim
            norm_px_acc = px_acc / char_dim
            
            gx, gy = k.ground_pos if k else (0, 0)
            is_ground_valid = (abs(gx) <= 150.0 and -50.0 <= gy <= 250.0)
            
            print(f"     Track {tid} ({tr.class_label}): age={tr.track_age}, bbox=({tr.current_bbox_xywh[0]:.1f}, {tr.current_bbox_xywh[1]:.1f}, {w:.1f}, {h:.1f})")
            print(f"       Ground: ({gx:.1f}, {gy:.1f}) -> is_ground_valid: {is_ground_valid}")
            print(f"       Pixel spd: {px_spd:.1f} px/s ({norm_px_spd:.2f} body/s), acc: {px_acc:.1f} px/s^2 ({norm_px_acc:.2f} body/s^2)")
            if k:
                print(f"       Cache spd: {k.speed_kmh:.1f} km/h | accel: {k.acceleration_mag:.1f} m/s^2")
        
        if len(tids) == 2:
            tr1, tr2 = tracks.get(tids[0]), tracks.get(tids[1])
            if tr1 and tr2:
                from scripts.benchmark_iteration_b_detector_recovery import compute_iou
                iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
                print(f"       IoU: {iou:.3f}")
