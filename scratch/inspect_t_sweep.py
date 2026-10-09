import json

with open('artifacts/t_full_benchmark_sweep.json') as f:
    d = json.load(f)

print('Keys:', list(d.keys()))
for r in d['per_video_results']:
    if r.get('fp_confirmed_count', 0) > 0:
        print('FP:', r['video_id'], r['split_type'], r['fp_confirmed_count'])
