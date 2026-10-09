import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

print('All videos in post_ab sweep:')
for i, p in enumerate(data['per_video_results']):
    print(f"{i+1}: {p['video_id']} | {p['split_type']} | TP={p.get('is_tp')} | FP={p['fp_confirmed_count']}")
