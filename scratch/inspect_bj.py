import json
data = json.load(open('artifacts/post_ab_full_benchmark_sweep.json'))
for r in data['per_video_results']:
    if r['video_id'] == 'bJMc3sNP2AU_00':
        print(r['split_type'], r['classification'], 'GT:', r['gt_frame'], 'Matched:', r['matched_candidate'])
        print('  FailureGate:', r.get('failure_gate'))
        print('  VLM Result:', r.get('vlm_tp_result'))
        print('  Episodes count:', r.get('consolidated_episodes_count'))
        print('  Raw cand count:', r.get('raw_candidates_count'))
        print('  FP conf:', r.get('fp_confirmed_count'), 'Unconf:', r.get('candidate_unconfirmed_count'))
