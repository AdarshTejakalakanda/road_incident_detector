import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        mat = p.get('matched_candidate', {})
        sigs = mat.get('signals', {})
        print(vid, 'post_event_rest=', sigs.get('post_event_rest'), 'S_rest=', sigs.get('S_rest'), 'Stationary=', sigs.get('stationary_duration_sec'))
