import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    print(p['video_id'], p['weather'], p['day_time'], 'FP =', p['fp_confirmed_count'])
