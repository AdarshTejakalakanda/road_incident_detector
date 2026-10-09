import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00':
        mat = p.get('matched_candidate', {})
        print('mat signals post_event_rest:', mat.get('signals', {}).get('post_event_rest'))
        print('mat signals S_rest:', mat.get('signals', {}).get('S_rest'))
