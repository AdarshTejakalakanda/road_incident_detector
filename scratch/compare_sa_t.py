import json

with open('artifacts/sa_full_benchmark_sweep.json') as f:
    sa = json.load(f)
with open('artifacts/t_full_benchmark_sweep.json') as f:
    t = json.load(f)

sa_vids = { (r['video_id'], r['split_type']): r for r in sa['per_video_results'] }
t_vids = { (r['video_id'], r['split_type']): r for r in t['per_video_results'] }

print(f"Total in SA: {len(sa_vids)}, Total in T: {len(t_vids)}")
print(f"SA TP: {sa['overall_true_positives']}, T TP: {t['overall_true_positives']}")
print(f"SA FN: {sa['overall_false_negatives']}, T FN: {t['overall_false_negatives']}")
print(f"SA FP: {sa['overall_false_positives']}, T FP: {t['overall_false_positives']}")
print(f"SA Recall: {sa['incident_recall']:.4f}, T Recall: {t['incident_recall']:.4f}")
print(f"SA Precision: {sa['precision']:.4f}, T Precision: {t['precision']:.4f}")
print(f"SA F1: {sa['f1_score']:.4f}, T F1: {t['f1_score']:.4f}")

print("\n=== Per-video differences ===")
for k in sa_vids:
    r_sa = sa_vids[k]
    r_t = t_vids.get(k, {})
    tp_sa = r_sa.get('is_tp')
    tp_t = r_t.get('is_tp')
    fp_sa = r_sa.get('fp_confirmed_count', 0)
    fp_t = r_t.get('fp_confirmed_count', 0)
    if tp_sa != tp_t or fp_sa != fp_t:
        print(f"\n{k[0]} ({k[1]}):")
        print(f"   TP: SA={tp_sa} -> T={tp_t}")
        print(f"   FP: SA={fp_sa} -> T={fp_t}")
        print(f"   Failure Gate SA: {r_sa.get('failure_gate')} -> T: {r_t.get('failure_gate')}")
        print(f"   Trig Frame SA: {r_sa.get('matched_candidate', {}).get('frame') if r_sa.get('matched_candidate') else None} -> T: {r_t.get('matched_candidate', {}).get('frame') if r_t.get('matched_candidate') else None}")
        print(f"   VLM calls SA: {r_sa.get('vlm_calls')} -> T: {r_t.get('vlm_calls')}")
