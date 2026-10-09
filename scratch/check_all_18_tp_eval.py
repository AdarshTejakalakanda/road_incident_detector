import json, sys, os
sys.path.insert(0, os.getcwd())
from modules.vlm_arbitrator import GroqArbitrator
arb = GroqArbitrator(api_key=None, mock_fallback=True)

with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

print('Evaluating all 18 TPs under new bypass logic:')
for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        cand = p['matched_candidate']
        if not cand:
            print(vid, 'No matched candidate!')
            continue
        sigs = cand["signals"]
        tel = {
            'event_type': cand['event_type'],
            'accel_ms2': sigs.get('accel_1_ms2', sigs.get('metric_accel_ms2', -60.0)),
            'bbox_iou': sigs.get('bbox_iou', 0.0),
            'distance_m': sigs.get('distance_m', 1.0),
            'norm_pixel_dist': sigs.get('norm_pixel_dist', 99.0),
            'S_dissip': sigs.get('S_dissip', 0.0),
            'post_event_rest': sigs.get('post_event_rest', False),
            'tau_rest': 0.25 if sigs.get('post_event_rest') else 0.0,
            'track_age': 10,
            'is_single_vehicle': cand['event_type'] in ('single_vehicle_anomaly', 'single'),
            'delTa_v_drop': 20.0 if sigs.get('S_v_drop', 0.0) > 0 else 0.0,
        }
        is_bp, ctype, btel, rat = arb._evaluate_deterministic_high_confidence_bypass(tel, cand['frame'], vid)
        print(vid, cand['frame'], 'Bypass=', is_bp, btel.get('bypass_reason'))
