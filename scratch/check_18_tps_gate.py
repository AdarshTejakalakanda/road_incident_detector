import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        cand = p.get('matched_candidate', {})
        sigs = cand.get('signals', {})
        a1 = sigs.get('accel_1_ms2')
        a2 = sigs.get('accel_2_ms2')
        acc = min(a1, a2) if (a1 is not None and a2 is not None) else (a1 if a1 is not None else a2)
        dist = sigs.get('distance_m', 1.0)
        norm_px = sigs.get('norm_pixel_dist', 99.0)
        iou = sigs.get('bbox_iou', 0.0)
        s_diss = sigs.get('S_dissip', 0.0)
        post_rest = sigs.get('post_event_rest', False)
        rest_satisfied = bool(post_rest)
        
        # Canonical match:
        has_contact = (dist <= 2.0 and iou >= 0.15) or (norm_px <= 0.50 and iou >= 0.30) or (iou >= 0.25)
        has_acc = (acc is not None and acc <= -12.0)
        has_diss = (s_diss >= 0.70 or post_rest)
        
        passes = has_contact and has_acc and has_diss and rest_satisfied
        print(vid, 'passes=', passes, 'has_contact=', has_contact, 'has_acc=', has_acc, 'acc=', acc, 'iou=', iou)
