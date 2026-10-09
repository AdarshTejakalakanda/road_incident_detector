import json
data = json.load(open('artifacts/post_ab_full_benchmark_sweep.json'))
targets = ['bJMc3sNP2AU_00', 'ftk_fSc4haM_00', 'UarP8qU1S-c_01', 'OXVXsdDT9QQ_00', 'Z4kg2Ev3vhk_00']
for t in targets:
    print('=== ' + t + ' ===')
    for r in data['per_video_results']: 
        if r['video_id'] == t:
            print(r['split_type'], r['classification'], 'GT:', r['gt_frame'], 'Matched:', r['matched_candidate'])
            print('  FailureGate:', r.get('failure_gate'))
            print('  VLM Result:', r.get('vlm_tp_result'))
            print('  Episodes count:', r.get('consolidated_episodes_count'))
            print('  Raw cand count:', r.get('raw_candidates_count'))
            print('  FP conf:', r.get('fp_confirmed_count'), 'Unconf:', r.get('candidate_unconfirmed_count'))
