import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        cand = p['matched_candidate']
        res = p.get('vlm_tp_result', {})
        btel = res.get('bypass_telemetry', {})
        reas = btel.get('bypass_reason', '')
        if 'stage_t_deterministic_bypass' not in reas:
            sigs = cand["signals"]
            print(vid, reas, 'accel=', sigs.get('accel_1_ms2'), 'iou=', sigs.get('bbox_iou'), 'dist=', sigs.get('distance_m'))
