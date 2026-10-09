import json

d = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', encoding='utf-8'))

print('Total evaluated:', d.get('total_evaluated'))
print('TP:', d.get('overall_true_positives'))
print('FN:', d.get('overall_false_negatives'))
print('FP:', d.get('overall_false_positives'))
print('Recall:', d.get('incident_recall'))
print('Precision:', d.get('precision'))
print('F1:', d.get('f1_score'))
print('Candidate unconfirmed:', d.get('total_candidate_unconfirmed'))
print('Bypass activations:', d.get('total_bypass_activations'))
print('Actor cooldown suppressed:', d.get('total_actor_cooldown_suppressed'))

tps = [x['video_id'] for x in d['per_video_results'] if x.get('is_tp')]
fns = [x['video_id'] for x in d['per_video_results'] if not x.get('is_tp')]
print('TP clips (18):', tps)
print('FN clips (12):', fns)