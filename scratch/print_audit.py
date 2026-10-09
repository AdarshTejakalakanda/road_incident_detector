import json

with open('scratch/fn_dispatches_audit.json') as f:
    data = json.load(f)

for vid_entry in data:
    vid = vid_entry['video_id']
    split = vid_entry['split']
    gt = vid_entry['gt_frame']
    eps = vid_entry['episodes']
    dispatched_eps = [e for e in eps if e.get('vlm_dispatched')]
    in_win_calls = sum(1 for ep in dispatched_eps if ep['is_in_gt_window'])
    out_win_calls = sum(1 for ep in dispatched_eps if not ep['is_in_gt_window'])
    print(f"\n=== {vid} ({split}, GT=F{gt}) ===")
    print(f"  Total episodes: {len(eps)}, Total VLM dispatches: {len(dispatched_eps)}")
    print(f"  In-window calls (within +-35 frames): {in_win_calls}")
    print(f"  Out-of-window calls: {out_win_calls}")
    for ep in eps:
        disp_str = "DISPATCHED" if ep['vlm_dispatched'] else "COOLDOWN_SUPPRESSED"
        in_win_flag = "IN_WIN" if ep['is_in_gt_window'] else "OUT_WIN"
        print(f"    Ep {ep['episode_id']} @ F{ep['peak_frame']} (delta={ep['delta_gt']:+d}, {in_win_flag}) | {disp_str} | tracks={ep['track_ids']} | type={ep['event_type']} | verdict={ep['vlm_verdict']}")
