import json

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    d = json.load(f)

cohort_c = []
cohort_d = []
cohort_a = []

for r in d['per_video_results']:
    vid = r['video_id']
    split = r['split_type']
    gt = r['gt_frame']
    cls = r['classification']
    fp_cnt = r['fp_confirmed_count']
    if cls == 'TP':
        mc = r.get('matched_candidate')
        cohort_c.append((vid, split, gt, mc))
    elif cls == 'FNG' or cls == 'FN':
        cohort_d.append((vid, split, gt, r.get('failure_gate'), r.get('raw_candidates_count'), r.get('vlm_calls')))
    if fp_cnt > 0:
        cohort_a.append((vid, split, fp_cnt, r.get('day_time'), r.get('out_of_window_episodes_count')))

print('Cohort C (TPs)', len(cohort_c))
for x in cohort_c:
    src = x[3].get('source') if x[3] else None
    frame = x[3].get('frame') if x[3] else None
    etype = x[3].get('event_type') if x[3] else None
    print('  TP:', x[0], x[1], 'gt:', x[2], 'frame:', frame, 'src:', src, 'etype:', etype)

print('\nCohort D (FNs)', len(cohort_d))
for x in cohort_d:
    print('  FN:', x[0], x[1], 'gt:', x[2], 'gate:', x[3], 'raw_cands:', x[4], 'vlm_calls:', x[5])

print('\nCohort A (FPss', len(cohort_a))
for x in cohort_a:
    print('  FP:', x[0], x[1], 'fp_cnt:', x[2], 'day_time:', x[3])
