import sys, os
sys.path.insert(0, os.path.abspath('.'))
from scratch.classify_fps_and_tps import fp_records
for i in range(5):
    r = fp_records[i]
    print(f"FP #{i+1}: {r['video_id']} ({r['split']}) @ f={r['trigger_frame']} (GT={r['gt_frame']}, diff={r['diff_frames']:+d}) dets={r['detectors']} tids={r['track_ids']}")
    if r["frame_data"]:
        f_idx, pts_ns, tracks, kin, conv = r["frame_data"]
        for tid in r["track_ids"]:
            tr = tracks.get(tid)
            k = kin.get(tid)
            if tr:
                bx, by, bw, bh = tr.current_bbox_xywh
                gc = tr.calculate_ground_contact(tr.current_bbox_xywh)
                print(f"      Track {tid} ({tr.class_label}): age={tr.track_age}, bbox=({bx:.1f}, {by:.1f}, {bw:.1f}, {bh:.1f}), gc_px=({gc[0]:.1f}, {gc[1]:.1f})")
            if k:
                print(f"        Kin: ground=({k.ground_pos[0]:.1f}, {k.ground_pos[1]:.1f}), spd={k.speed_kmh:.1f}km/h, a_mag={k.acceleration_mag:.1f}m/s2")
