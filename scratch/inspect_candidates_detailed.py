import os, sys, pickle, json
sys.path.insert(0, '.')
from modules.vlm_arbitrator import passes_tier1_admission

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    post = json.load(f)

print('Post-AB quarantined events summary:')
for x in post['per_video_results']:
    unc = x.get('candidate_unconfirmed_count', 0)
    if unc > 0:
        print(f"{x['video_id']} | {x['split_type']}: unconfirmed={unc}, vlm_calls={x.get('vlm_calls')}")
