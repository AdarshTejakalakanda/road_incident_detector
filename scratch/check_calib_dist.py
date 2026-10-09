import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        mat = p.get('matched_candidate', {})
        sigs = mat.get('signals', {})
        dist = sigs.get('distance_m')
        iou = sigs.get('bbox_iou')
        acc = sigs.get('accel_1_ms2')
        print(vid, dist, iou, acc)
