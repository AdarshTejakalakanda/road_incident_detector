import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    post = json.load(f)

for p in post['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00' and p['is_tp']:
        print('Bgd Fields:')
        print('matched_candidate:', p.get('matched_candidate'))
        print('vlm_tp_result:', p.get('vlm_tp_result'))
