import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

for x in d['per_video_results']:
    if x.get('is_fn'):
        print(f"FN: {x['video_id']} | {x['split_type']} | gt={x.get('gt_frame')} | gate={x.get('failure_gate')} | raw_cands={x.get('raw_candidates_count')} | in_window_eps={x.get('in_window_episodes_count')} | vlm_calls={x.get('vlm_calls')}")
