import json

d = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', encoding='utf-8'))
pvr = d['per_video_results']

noc_clips = [x for x in pvr if x.get('day_time') == 'night']

total_noc_fp = 0
out = []
for x in noc_clips:
    vid = x['video_id']
    split = x['split_type']
    tp = x['is_tp']
    fp = x['fp_confirmed_count']
    total_noc_fp += fp
    out.append(vid.ljust(18) + ' | ' + split.ljust(22) + ' | TP: ' + str(tp).ljust(5) + ' | FP: ' + str(fp).rjust(2))

out.append('Total Nocturnal FP count: ' + str(total_noc_fp))
with open('scratch/nocturnal_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))