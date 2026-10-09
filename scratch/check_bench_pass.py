import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        res = p.get('vlm_tp_result', {})
        brew = res.get('bypass_telemetry', {})
        vel = brew.get('velocities', {})
        rest = brew.get('rest_and_arrest', {})
        print(vid, 'accel=', brew.get('impulse', {}).get('accel_ms2'), 'vels=', vel, 'rest=', rest)
