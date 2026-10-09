import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

for x in d['per_video_results']:
    if x['classification']=='TP':
        print(x['video_id'], x['split_type'], 'vlm_tp_result:', x['vlm_tp_result'].get('source') if x['vlm_tp_result'] else None, x['vlm_tp_result'].get('verdict') if x['vlm_tp_result'] else None)
