import json

with open('artifacts/stage_t_remediated_benchmark_sweep.json') as f:
    d = json.load(f)

pvr = d.get('per_video_results', [])
target_vids = ['kABWTDKgc9U_00', 'ftk_fSc4haM_00', 'BgdKGnFqz0I_00', 'iA_u73BG4ZM_00', 'maJGw3hQs_s_00', 'aF2Hbp8C23A_00', 'LiM5LPSZoow_00']

seen = set()
for v in pvr:
    vid = v['video_id']
    if vid in target_vids and vid not in seen:
        seen.add(vid)
        mc = v.get('matched_candidate') or {}
        print(f"{vid}: gt={v.get('gt_frame')}, type={v.get('incident_type')}, day={v.get('day_time')}, "
              f"cand_frame={mc.get('frame')}, tracks={mc.get('track_ids')}, etype={mc.get('event_type')}")
