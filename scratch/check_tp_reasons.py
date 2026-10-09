import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

print("TP details in prior sweep:")
for p in data['per_video_results']:
    if p['is_tp']:
        res = p['vlm_tp_result']
        brew = res.get('bypass_telemetry', {})
        print(p['video_id'], 'G' + str(p['gt_frame']), brew.get('bypass_reason'), brew.get('contact_evidence'), brew.get('velocities'), brew.get('rest_and_arrest'))
