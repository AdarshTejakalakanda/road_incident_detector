import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import pickle
import numpy as np

with open("telemetry_phase1_final_validation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vids_meta = {v["video_id"]: v for v in data["all_videos"]}

# Load videos from caches
all_cached_videos = {}
for split in ["in_distribution", "geo_aware"]:
    with open(f"cache_videos_iter_b_split_{split}_15.pkl", "rb") as f:
        vids = pickle.load(f)
    for v in vids:
        all_cached_videos[(split, v["video_id"])] = v

print("=================================================================")
print("PART 1: THE 12 TRUE POSITIVE EPISODES (MUST PRESERVE 100%)")
print("=================================================================")
tp_list = []
for v in data["all_videos"]:
    if v["status"] == "TP":
        tp_list.append(v)
        vid_id = v["video_id"]
        split = v["split"]
        v_cached = all_cached_videos.get((split, vid_id))
        cand_f = v["cand_frame"]
        gt_f = v["gt_frame"]
        fps = v["lead_sec"] # or from cached
        print(f"TP Video: {vid_id} ({split}) | GT={gt_f}, Cand={cand_f}, Diff={v['diff_frames']:+d} ({v['lead_sec']:+.2f}s) | Primary={v['primary_detector']}")
        # Find candidates within +/- 35 frames
        cands_tp = [c for c in v["candidates"] if abs(c["f"] - gt_f) <= 35]
        for c in cands_tp[:3]:
            print(f"   f={c['f']} det={c['det']} tids={c['tids']}")

print(f"\nTotal TPs: {len(tp_list)}")

print("\n=================================================================")
print("PART 2: THE 22 FALSE POSITIVE EPISODES (CLASSIFICATION BY CAUSE)")
print("=================================================================")
fp_records = []
for v in data["all_videos"]:
    if v["fp_episodes"] > 0:
        vid_id = v["video_id"]
        split = v["split"]
        v_cached = all_cached_videos.get((split, vid_id))
        gt_f = v["gt_frame"]
        
        # Candidate episodes (diff > 35)
        # Cluster the raw candidates to get the exact episodes
        from scripts.benchmark_iteration_b_detector_recovery import cluster_episodes
        raw_cands = [c for c in v["candidates"] if abs(c["f"] - gt_f) > 35]
        # Group into raw cands format for cluster_episodes
        # In candidates list: {'f': c['frame_idx'], 'tids': c['track_ids'], 'det': c.get('detector')}
        c_for_cluster = [{'frame_idx': c['f'], 'track_ids': c['tids'], 'detector': c['det']} for c in raw_cands]
        eps = cluster_episodes(c_for_cluster, max_gap_frames=25)
        
        for ep_idx, ep in enumerate(eps):
            trig_f = ep["trigger_frame"]
            diff = trig_f - gt_f
            rep = ep["representative"]
            dets = [rep.get("detector", "unknown")]
            tids = rep.get("track_ids", [])
            
            # Find the frame data
            fr_data = next((f for f in v_cached["frames"] if f[0] == trig_f), None)
            
            fp_records.append({
                "video_id": vid_id,
                "split": split,
                "gt_frame": gt_f,
                "trigger_frame": trig_f,
                "diff_frames": diff,
                "detectors": dets,
                "track_ids": tids,
                "cand_count": ep["raw_count"],
                "frame_data": fr_data,
            })

print(f"Total FP episodes clustered: {len(fp_records)}")
for i, r in enumerate(fp_records, 1):
    diff_s = r["diff_frames"] / 30.0 # approx
    print(f"\nFP #{i}: Video {r['video_id']} ({r['split']}) @ f={r['trigger_frame']} (GT={r['gt_frame']}, diff={r['diff_frames']:+d}f)")
    print(f"    Detectors: {r['detectors']} | Track IDs: {r['track_ids']} | Alarms: {r['cand_count']}")
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
                if len(tr.position_history) >= 2:
                    p1 = tr.position_history[-2]
                    p2 = tr.position_history[-1]
                    p_disp = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
                    print(f"        Px disp last frame: {p_disp:.2f} px")
