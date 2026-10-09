import json

with open('artifacts/t_full_benchmark_sweep.json') as f:
    t = json.load(f)

for x in t['per_video_results']:
    if x.get('fp_confirmed_count', 0) > 0:
        print(f"t_sweep FP: {x['video_id']} | {x['split_type']} | fp_count={x.get('fp_confirmed_count')}")
