import json

with open('artifacts/stage_t_remediated_benchmark_sweep.json') as f:
    d = json.load(f)

pvr = d.get('per_video_results', [])
print(f"Total entries: {len(pvr)}")
tps = []
for v in pvr:
    if v.get('is_tp'):
        vid = v['video_id']
        gt = v.get('gt_frame')
        itype = v.get('incident_type')
        day = v.get('day_time')
        matched = v.get('matched_candidate')
        tps.append((vid, gt, itype, day, matched))
        print(f"TP: {vid} | gt={gt} | type={itype} | day={day} | cand={matched}")

print(f"\nTotal TPs: {len(tps)}")
