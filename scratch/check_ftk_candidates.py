import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'ftk_fSc4haM_00':
        cand = p.get('matched_candidate', {})
        print('ftk signals:', cand.get('signals', {}).get('accel_1_ms2'), cand.get('signals', {}).get('accel_2_ms2'))
