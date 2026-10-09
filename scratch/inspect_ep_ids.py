import json
data = json.load(open('artifacts/post_ab_full_benchmark_sweep.json'))
targets = ['bJMc3sNP2AU_00', 'ftk_fSc4haM_00', 'UarP8qU1S-c_01', 'OXVXsdDT9QQ_00', 'Z4kg2Ev3vhk_00']
for t in targets:
    for r in data['per_video_results']:
        if r['video_id'] == t:
            print(t, r['split_type'])
            for ep in r.get('episodes', []):
                if 'ConsolidatedEpisode' in ep:
                    parts = ep.split(',')[0]
                    print('  ', parts)
