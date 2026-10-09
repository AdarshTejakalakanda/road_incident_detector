import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'LiM5LPSZoow_00':
        cand = p.get('matched_candidate', {})
        print('LiM5 signals:', cand.get('signals', {}))
