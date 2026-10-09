import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] in ('Z4kg2Ev3vxk_00', 'ftk_fSc4haM_00', 'UarP8qU1S-c_00'):
        print(p['video_id'], 'FP =', p['fp_confirmed_count'])
