import json
with open('artifacts/sa_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

print([(per['video_id'], per['is_tp']) for per in data['per_video_results']])
