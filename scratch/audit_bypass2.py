import json
d = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', encoding='utf-8'))
pvr = d['per_video_results'][0]
print('Keys in per_video_result:', list(pvr.keys()))
print('Sample pvr:', pvr['video_id'], pvr['is_tp'], pvr['fp_count'], pvr['pred_frames'])