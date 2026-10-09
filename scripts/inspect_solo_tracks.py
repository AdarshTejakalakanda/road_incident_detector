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

for vid in ['nAXTthLfgtI_00', 'vxZ47DoPNY8_00', 'VozrseCzhQc_00', 'UarP8qU1S-c_00', 'UarP8qU1S-c_01']:
    v = vids[vid]
    print(f"\n==========================================")
    print(f"Video: {vid}, GT Frame: {v['gt_frame']}")
    print(f"==========================================")
    all_tracks = set()
    frames_with_tracks = []
    for f_idx, pts, tracks, kin, conv in v['frames']:
        if tracks:
            frames_with_tracks.append(f_idx)
            for tid, tr in tracks.items():
                all_tracks.add((tid, tr.class_label))
    print(f"Total frames: {len(v['frames'])}")
    print(f"Frames with tracks: {len(frames_with_tracks)}: min={min(frames_with_tracks) if frames_with_tracks else None}, max={max(frames_with_tracks) if frames_with_tracks else None}")
    print(f"All tracks seen: {all_tracks}")
    
    # Print tracks around GT frame +/- 35
    for f_idx, pts, tracks, kin, conv in v['frames']:
        if abs(f_idx - v['gt_frame']) <= 35 and tracks:
            t_str = ", ".join([f"id={tid}({tr.class_label}, conf={tr.detection_confidence:.2f}, box={[round(x,1) for x in tr.current_bbox_xywh]})" for tid, tr in tracks.items()])
            print(f"  f={f_idx}: {t_str}")
