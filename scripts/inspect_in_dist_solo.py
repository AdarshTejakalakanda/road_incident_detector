import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle

with open('cache_videos_iter_b_split_in_distribution_15.pkl', 'rb') as f:
    in_dist = pickle.load(f)

vids = {v['video_id']: v for v in in_dist}
for vid in ['nAXTthLfgtI_00', 'vxZ47DoPNY8_00', 'VozrseCzhQc_00']:
    v = vids[vid]
    all_tracks = set()
    frames_with_tracks = []
    for f_idx, pts, tracks, kin, conv in v['frames']:
        if tracks:
            frames_with_tracks.append(f_idx)
            for tid, tr in tracks.items():
                all_tracks.add((tid, tr.class_label))
    print(f"=== {vid} (GT={v['gt_frame']}) ===")
    print(f"Total frames={len(v['frames'])}, frames_with_tracks={len(frames_with_tracks)}")
    print(f"All tracks: {all_tracks}")
    for f_idx, pts, tracks, kin, conv in v['frames']:
        if abs(f_idx - v['gt_frame']) <= 35 and tracks:
            print(f"  f={f_idx}: {[(tid, tr.class_label, [round(x,1) for x in tr.current_bbox_xywh]) for tid, tr in tracks.items()]}")
