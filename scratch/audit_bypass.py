import json
from collections import Counter

d = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', encoding='utf-8'))

reasons = Counter()
by_video = {}

for x in d['per_video_results']:
    vid = x['video_id']
    fps = []
    for ep in x.get('episodes', []):
        # check if episode is FP
        gt = x.get('gt_frame')
        peak = ep.get('peak_frame')
        # in benchmark, if abs(peak - gt) > 35 or gt is None, it is FP
        is_near_gt = (gt is not None) and (abs(peak - gt) <= 35)
        # However, multiple detections in window: only first or closest is TP
        verdict = ep.get('verdict')
        source = ep.get('source')
        bt = ep.get('bypass_telemetry', {})
        reason = bt.get('bypass_reason', ep.get('rationale', 'unknown'))
        reasons[reason] += 1
        fps.append({'frame': peak, 'source': source, 'reason': reason, 'is_near_gt': is_near_gt})
    by_video[vid] = fps

print('TOP BYPASS REASONS:')
for r, c in reasons.most_common(10):
    print(f'  {c:3d}: {r}')
