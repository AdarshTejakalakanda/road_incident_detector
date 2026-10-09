import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00' and p['is_tp']:
        cand = p['matched_candidate']
        sigs = cand['signals']
        print('Bgd signals: dpos=', sigs.get('distance_m'), 'drop=', sigs.get('delta_v_drop'), 'v_drop=', sigs.get('S_v_drop'), 'diss=', sigs.get('S_dissip'), 'rest=', sigs.get('post_event_rest'))
