import json
data = json.load(open('artifacts/post_ab_full_benchmark_sweep.json'))
for r in data['per_video_results']:
    if r['video_id'] == 'bJMc3sNP2AU_00':
        print(list(r.keys()))
