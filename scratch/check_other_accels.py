import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        cand = p.get('matched_candidate', {})
        sigs = cand.get('signals', {})
        p1, d = sigs.get('accel_1_ms2'), sigs.get('accel_2_ms2')
        print(vid, 'a1cel=', p1, 'a2cel=', d)
