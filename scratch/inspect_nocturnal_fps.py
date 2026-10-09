import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

noct_vids = []
noct_fps = 0
for x in d['per_video_results']:
    if x.get('day_time') == 'night' or x.get('weather') == 'night':
        fp_c = x.get('fp_confirmed_count', 0)
        noct_vids.append((x['video_id'], x['split_type'], fp_c))
        noct_fps += fp_c

print('Nocturnal videos evaluated:', len(noct_vids))
for v in noct_vids:
    print('  ', v[0], v[1], 'fps:', v[2])
print('Absolute Nocturnal FP Count:', noct_fps)
