import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00':
        mat = p.get('matched_candidate', {})
        sigs = mat.get('signals', {})
        print('signals keys:(', list(sigs.keys()))
        print('post_event_rest:', sigs.get('post_event_rest'))
        print('tau_rest:', sigs.get('tau_rest'))
        print('rest_duration_s:', sigs.get('rest_duration_s'))
        print('stationary_duration_sec:', sigs.get('stationary_duration_sec'))
