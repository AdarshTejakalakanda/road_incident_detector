import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

for x in d['per_video_results']:
    if x.get('fp_confirmed_count', 0) > 0:
        print(x['video_id'], x['split_type'], 'fp_confirmed:', x['fp_confirmed_count'], 'day_time:', x.get('day_time'))
        for ep in x.get('out_of_window_episodes', []):
            print('   episode:', ep)
