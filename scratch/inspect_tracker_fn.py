import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle

for split in ["split_in_distribution", "split_geo_aware"]:
    with open(f"cache_videos_iter_b_{split}_15.pkl", "rb") as f:
        vids = pickle.load(f)
    for v in vids:
        if v["video_id"] in ["fzWY0vLAXzI_00", "kABWTDKgc9U_00"]:
            print(f"\nVideo: {v['video_id']} ({split}) | GT: {v['gt_frame']}")
            # Look at tracks present in [GT-15..GT+15]
            track_life = {}
            for f_idx, pts, tr, kin, conv in v["frames"]:
                if abs(f_idx - v["gt_frame"]) <= 15:
                    for tid, t in tr.items():
                        if tid not in track_life:
                            track_life[tid] = {"frames": [], "label": t.class_label, "confs": []}
                        track_life[tid]["frames"].append(f_idx)
                        track_life[tid]["confs"].append(t.detection_confidence)
            print(f"Active tracks in [GT-15..GT+15]: {len(track_life)}")
            for tid, data in track_life.items():
                print(f"  Track {tid:3d}: {data['label']:10s} | frames={len(data['frames'])} [{min(data['frames'])}..{max(data['frames'])}] | mean_conf={sum(data['confs'])/len(data['confs']):.2f}")
