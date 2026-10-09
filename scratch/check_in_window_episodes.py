import json
with open('artifacts/sa_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    print(p['video_id'], 'in_win:', p['candidates_count'] if 'candidates_count' in p else p['raw_candidates_count'], 'in_win_eps:', p['in_window_episodes_count'], 'attrib:', p['vlm_tp_result'])
