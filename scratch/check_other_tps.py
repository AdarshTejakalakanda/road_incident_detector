import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] in ('unS0-TLF1ao_00', 'fzWY0vLAXzI_00', 'vxZ47DoPNY8_00', 'ftk_fSc4haM_00', 'LiM5LPSZoow_00', 'iA_u73BG4ZM_00', 'maJGw3hQs_s_00', 'aF2Hbp8C23A_00', 'F0NHENn-HS4_00'):
        res = p.get('vlm_tp_result', {})
        btel = res.get('bypass_telemetry', {})
        print(p['video_id'], res.get('source'), btel.get('bypass_reason'))
