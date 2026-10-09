import json

with open('artifacts/r_full_benchmark_sweep.json') as f:
    d = json.load(f)

print(f"Total TPs: {d.get('overall_true_positives')}")
for v in d.get('per_video_results', []):
    if v.get('is_tp'):
        cand = v.get('matched_candidate') or {}
        sigs = cand.get('signals', {})
        res = v.get('vlm_tp_result') or {}
        print(f"{v['video_id']} ({v['split_type']}): F{cand.get('frame')} {cand.get('event_type')}")
        print(f"   sigs: accel={sigs.get('metric_accel_ms2', sigs.get('accel_1_ms2'))}, dist={sigs.get('ground_distance_m', sigs.get('distance_m'))}, iou={sigs.get('bbox_iou')}, theta={sigs.get('yaw_delta_deg', sigs.get('theta_deg'))}, speed={sigs.get('metric_speed_kmh', sigs.get('speed_kmh'))}, s_single={sigs.get('s_single')}, S_dissip={sigs.get('S_dissip')}")
        print(f"   res: {res.get('verdict')} / {res.get('source')} / {res.get('collision_type')}")
