import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

night_fps = 0
for p in data['per_video_results']:
    if p['weather'] == 'normal' and p['day_time'] == 'night':
        night_fps += p['fp_confirmed_count']
        print(p['video_id'], 'FP=', p['fp_confirmed_count'])

print("Total night fps in prior sweep:", night_fps)
