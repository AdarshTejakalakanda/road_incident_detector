import json, os, sys
sys.path.insert(0, '.')
from modules.vlm_arbitrator import passes_tier1_admission

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

print('total_candidate_unconfirmed:g', d.get('total_candidate_unconfirmed'))
