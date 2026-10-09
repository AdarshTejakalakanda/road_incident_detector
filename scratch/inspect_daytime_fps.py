import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

print('Daytime FPs across videos:')
for x in d['per_video_results']:
    day_t = x.get('day_time', '')
    fp_c = x.get('fp_confirmed_count', 0)
    if fp_c > 0:
        print(f"{x['video_id']} | {x['split_type']} | day_time={day_t} | fp_count={fp_c} | out_eps={x.get('out_of_window_episodes_count')}")
