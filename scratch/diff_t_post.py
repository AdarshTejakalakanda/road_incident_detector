import json
with open('artifacts/t_full_benchmark_sweep.json') as f:
    t = json.load(f)
with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    post = json.load(f)

t_dict = {(x['video_id'], x['split_type']): x for x in t['per_video_results']}
post_dict = {(x['video_id'], x['split_type']): x for x in post['per_video_results']}

for k, t_res in t_dict.items():
    p_res = post_dict.get(k, {})
    t_tp = t_res.get('is_tp')
    p_tp = p_res.get('is_tp')
    t_fp = t_res.get('fp_confirmed_count')
    p_fp = p_res.get('fp_confirmed_count')
    if t_tp != p_tp or t_fp != p_fp:
        print(k[0], k[1], 'TP:', t_tp, '->', p_tp, '| FP:', t_fp, '->', p_fp)
