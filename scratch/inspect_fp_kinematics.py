import sys
import os
sys.path.insert(0, os.path.abspath("."))
import pickle

with open("telemetry_phase1_final_validation.json", "r", encoding="utf-8") as f:
    import json
    data = json.load(f)

vids_meta = {v["video_id"]: v for v in data["all_videos"]}

for split in ["in_distribution", "geo_aware"]:
    print(f"\n=================== Split: {split} ===================")
    with open(f"cache_videos_iter_b_split_{split}_15.pkl", "rb") as f:
        vids = pickle.load(f)

    for v in vids:
        vid_id = v["video_id"]
        v_meta = vids_meta.get(vid_id)
        if not v_meta or v_meta["fp_episodes"] == 0:
            continue

        gt_f = v["gt_frame"]
        fps = v["fps"]

        # Find candidates for this video that are FPs (diff > 35 frames from GT)
        fp_cands = [c for c in v_meta["candidates"] if abs(c["f"] - gt_f) > 35]
        if not fp_cands:
            continue

        print(f"\n--- Video: {vid_id} (GT={gt_f}, fps={fps}, status={v_meta['status']}) ---")
        for c in fp_cands:
            cf = c["f"]
            tids = c["tids"]
            det = c["det"]
            diff = cf - gt_f

            # Find this frame in v['frames']
            frame_tuple = next((f for f in v["frames"] if f[0] == cf), None)
            if not frame_tuple:
                print(f"  Frame {cf} not found!")
                continue

            f_idx, pts_ns, tracks, kin, conv = frame_tuple
            print(f"  Candidate @ f={cf} (diff={diff:+d}f / {diff/fps:+.2f}s) det={det} tids={tids}:")

            for tid in tids:
                tr = tracks.get(tid)
                k = kin.get(tid)
                if tr:
                    bbox = tr.current_bbox_xywh
                    print(f"    Track {tid} ({tr.class_label}): age={tr.track_age}, bbox={bbox}")
                if k:
                    print(f"      Kinematics: ground_pos=({k.ground_pos[0]:.2f}, {k.ground_pos[1]:.2f}), speed_kmh={k.speed_kmh:.1f}, accel_mag={k.acceleration_mag:.1f}")
                    print(f"      Pixel Kin:  px_spd={getattr(k, 'pixel_speed', 0.0):.1f}, px_acc={getattr(k, 'pixel_acceleration', 0.0):.1f}, ground_valid={getattr(k, 'is_ground_valid', True)}")

            # Also check convergence if pair
            if len(tids) == 2:
                matching_conv = [cv for cv in conv if (cv.track_id_1 in tids and cv.track_id_2 in tids)]
                if matching_conv:
                    mc = matching_conv[0]
                    print(f"      Convergence: dist_m={mc.distance_m:.2f}, converging={mc.is_converging}, tca={mc.tca_sec:.2f}, dca={mc.dca_m:.2f}")
