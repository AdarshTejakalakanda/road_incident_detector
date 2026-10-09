import json

with open("artifacts/stage_t_cohort_validation.json") as f:
    cv = json.load(f)

cohort_c = cv["cohort_c_tp_retention"]["records"]
print(f"Cohort C records count: {len(cohort_c)}")

for i, rec in enumerate(cohort_c):
    print(f"Rec {i+1}: idx={rec.get('index')}, video={rec.get('video_id')}, gt_frame={rec.get('gt_frame')}, cand_frame={rec.get('candidate_frame')}, split={rec.get('split_type')}, type={rec.get('event_type')}, verdict={rec.get('verdict')}")
