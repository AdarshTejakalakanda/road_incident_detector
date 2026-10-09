import json

with open('artifacts/stage_t_remediated_benchmark_sweep.json') as f:
    d = json.load(f)

pvr = d.get('per_video_results', [])
for i, v in enumerate(pvr):
    if v.get('is_tp'):
        vid = v['video_id']
        gt = v.get('gt_frame')
        itype = v.get('incident_type')
        day = v.get('day_time')
        mc = v.get('matched_candidate', {})
        f_cand = mc.get('frame') if mc else None
        tracks = mc.get('track_ids') if mc else None
        etype = mc.get('event_type') if mc else None
        print(f"TP #{i+1}: vid={vid} | gt={gt} | type={itype} | day={day} | cand_frame={f_cand} | tracks={tracks} | etype={etype}")
