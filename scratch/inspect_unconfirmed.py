import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

total_unconf = 0
for r in d['per_video_results']:
    unconf = r.get('candidate_unconfirmed_count', 0)
    total_unconf += unconf
    if unconf > 0:
        print(f"{r['video_id']} ({r['split_type']}): unconf={unconf}, cls={r.get('classification')}, GT={r.get('gt_frame')}")
print('Total unconfirmed across all 30 runs:', total_unconf)
