import json, os, sys
sys.path.insert(0, '.')
from modules.vlm_arbitrator import passes_tier1_admission

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    post = json.load(f)

print('total_candidate_unconfirmed:', post.get('total_candidate_unconfirmed'))
