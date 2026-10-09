import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

for x in d['per_video_results']:
    if x['classification']=='TP':
        vid = x['video_id']
        split = x['split_type']
        mc = x['matched_candidate']
        vres = x['vlm_tp_result']
        print(f"{vid} | {split} | frame={mc.get('frame')} | src={vres.get('source')} | verdict={vres.get('verdict')} | rationale={vres.get('rationale')}")
