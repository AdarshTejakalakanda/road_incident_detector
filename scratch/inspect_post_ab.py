import json

with open('artifacts/post_ab_full_benchmark_sweep.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)

print('=== SUMMARY METRICS ===')
print('Overall TP:', bench['overall_true_positives'])
print('Overall FN:', bench['overall_false_negatives'])
print('Overall FP:', bench['overall_false_positives'])
print('Overall Recall:', bench['incident_recall'])
print('Overall Precision:', bench['precision'])
print('Overall F1:', bench['f1_score'])

day_fp = sum(r['fp_confirmed_count'] for r in bench['per_video_results'] if r.get('day_time') == 'day')
night_fp = sum(r['fp_confirmed_count'] for r in bench['per_video_results'] if r.get('day_time') == 'night')
print(f'Daytime FP: {day_fp}, Nocturnal FP: {night_fp}')
unconf = bench.get('total_candidate_unconfirmed')
bypass = bench.get('total_bypass_activations')
supp = bench.get('overall_fp_suppressed')
print(f'Total Candidate Unconfirmed: {unconf}')
print(f'Total Bypass Activations: {bypass}')
print(f'Total FP Suppressed: {supp}')

print('\n=== SPLIT BREAKDOWN ===')
for sname, smetrics in bench['split_metrics'].items():
    tp = smetrics['TP']
    fn = smetrics['FN']
    fp = smetrics['FP']
    total = tp + fn
    rec = tp / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f'{sname}:')
    print(f'  TP: {tp}, FN: {fn}, FP: {fp}')
    print(f'  Recall: {rec:.4f} ({tp}/{total})')
    print(f'  Precision: {prec:.4f}')
    print(f'  F1: {f1:.4f}')

target_cases = ['bJMc3sNP2AU_00', 'ftk_fSc4haM_00', 'UarP8qU1S-c_01', 'OXVXsdDT9QQ_00', 'Z4kg2Ev3vhk_00']
print('\n=== KEY RECOVERY & TARGET CASES ===')
for vid in target_cases:
    matches = [r for r in bench['per_video_results'] if r['video_id'] == vid]
    for m in matches:
        mc = m.get('matched_candidate')
        trig = mc.get('frame') if mc else None
        st = m['split_type']
        cl = m['classification']
        gt = m['gt_frame']
        vc = m['vlm_calls']
        fpc = m['fp_confirmed_count']
        unc = m['candidate_unconfirmed_count']
        fg = m.get('failure_gate')
        print(f'\nVideo: {vid} ({st})')
        print(f'  GT Frame: {gt}, Incident Type: {m.get(" incident_type\)}, Day/Night: {m.get(\day_time\)}')
 print(f' Classification: {cl}, Matched Frame: {trig}')
 print(f' Episode count: {m[\consolidated_episodes_count\]}, Raw cand count: {m[\raw_candidates_count\]}')
 print(f' VLM calls: {vc}, FP conf: {fpc}, Unconf: {unc}, Failure Gate: {fg}')
 if m.get('vlm_tp_result'):
 verd = m['vlm_tp_result'].get('verdict')
 src = m['vlm_tp_result'].get('source')
 rat = m['vlm_tp_result'].get('rationale')
 print(f' VLM Verdict: {verd}')
 print(f' VLM Source: {src}')
 print(f' VLM Rationale: {rat}')
 if m.get('episodes'):
 print(f' Episodes count: {len(m[\episodes\])}')
