import json

sweep = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', 'r', encoding='utf-8'))
for tp in sweep['per_video_results']:
    if tp['is_tp']:
        vid = tp['video_id']
        match = tp.get('matched_candidate') or {}
        sig = match.get('signals') or {}
        print(vid, 'signals keys:', list(sig.keys()))
        print('   s_v_drop:', sig.get('S_v_drop'), 'decel_mag:', sig.get('decel_mag'), 'metric_accel:', sig.get('metric_accel_ms2'), 'max_recent:', sig.get('max_recent_speed_kmh'), 'spd:', sig.get('speed_kmh'), 'kinematic_plausibility:', sig.get('kinematic_plausibility'))
        break