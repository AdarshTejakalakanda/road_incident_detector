import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

print('Total CANDIDATE_UNCONFIRMED:', d.get('total_candidate_unconfirmed'))


for x in d['per_video_results']:
    unconf = x.get('candidate_unconfirmed_count', 0)
    vlm_c = x.get('vlm_calls', 0)
    dps = x.get('out_of_window_episodes_count', 0)
    fpc = x.get('fp_confirmed_count', 0)
    vid = x['video_id']
    split = x['split_type']
    if unconf > 0 or vlm_c > 0 or dps > 0: 
        print(vid, split, 'unconf:', unconf, 'vlm_calls:', vlm_c, 'out_eps:', dps, 'fp:', fpc)
