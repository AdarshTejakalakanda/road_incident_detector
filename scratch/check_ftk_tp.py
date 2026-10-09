import json
with open('artifacts/sa_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'ftk_fSc4haM_00':
        print('ftk status:', p['is_tp'], p['is_fn'])
        res = p.get('vlm_tp_result', {})
        print('source:', res.get('source'), 'verdict:', res.get('verdict'))
        print('bypass_telemetry:', res.get('bypass_telemetry'))
