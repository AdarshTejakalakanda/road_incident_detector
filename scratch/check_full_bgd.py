import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00':
        print('Bgd keys:', list(p.keys()))
        print('vlm_tp_result:', p.get('vlm_tp_result'))
