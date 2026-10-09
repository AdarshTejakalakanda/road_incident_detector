import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00':
        print('Bgd total candidates:', p['total_candidates_generated'],' in_win:', p['in_window_episodes_count'], 'out_win:', p['out_of_window_episodes_count'])
