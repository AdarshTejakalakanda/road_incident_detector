import json

path_eval = "artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json"
path_supp = "artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json"

with open(path_eval, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

with open(path_supp, "r", encoding="utf-8") as f:
    supp_data = json.load(f)

# 1. Exact record count and unique episode IDs
eval_record_count = len(eval_data)
eval_ep_ids = [r["episode_id"] for r in eval_data]
eval_unique_ep_ids = set(eval_ep_ids)

supp_record_count = len(supp_data)
supp_ep_ids = [r["episode_id"] for r in supp_data]
supp_unique_ep_ids = set(supp_ep_ids)

print("=== 1. RECORD COUNTS & UNIQUE EPISODE IDS ===")
print(f"Evaluation artifact ({path_eval}):")
print(f"  Total records: {eval_record_count}")
print(f"  Unique episode IDs: {len(eval_unique_ep_ids)}")
print(f"  Duplicate episode IDs: {len(eval_ep_ids) - len(eval_unique_ep_ids)}")

print(f"\nSuppression artifact ({path_supp}):")
print(f"  Total records: {supp_record_count}")
print(f"  Unique episode IDs: {len(supp_unique_ep_ids)}")
print(f"  Duplicate episode IDs: {len(supp_ep_ids) - len(supp_unique_ep_ids)}")

# 2. Check whether the 16 quarantined episodes appear in the 53-record evaluation artifact
intersection = eval_unique_ep_ids.intersection(supp_unique_ep_ids)
print("\n=== 2 & 3. SET RELATIONSHIPS (INTERSECTION & DIFFERENCES) ===")
print(f"Intersection count (episodes present in BOTH files): {len(intersection)}")
print(f"Are all {supp_record_count} suppression episodes in evaluation artifact?: {intersection == supp_unique_ep_ids}")

# Set differences
diff_supp_minus_eval = supp_unique_ep_ids - eval_unique_ep_ids
diff_eval_minus_supp = eval_unique_ep_ids - supp_unique_ep_ids
print(f"Suppression - Evaluation (in suppression but NOT in evaluation): {len(diff_supp_minus_eval)} -> {diff_supp_minus_eval}")
print(f"Evaluation - Suppression (in evaluation but NOT in suppression): {len(diff_eval_minus_supp)}")

# List the intersection items
print("\nExact intersection episodes (sorted):")
for eid in sorted(list(intersection)):
    ev_match = [r for r in eval_data if r["episode_id"] == eid][0]
    su_match = [r for r in supp_data if r["episode_id"] == eid][0]
    print(f"  {eid}: eval_frame={ev_match['anchor_frame']}, eval_disp={ev_match['disposition']}, eval_score={ev_match['score']} | supp_frame={su_match['anchor_frame']}, supp_disp={su_match['disposition']}, supp_score={su_match['score']}")

# 4. Check F176
print("\n=== 4. F176 MEMBERSHIP CHECK ===")
f176_id = "5uh8-3q51uI_00_ep01_f24"
f176_frame = 182

f176_in_eval_by_id = [r for r in eval_data if r["episode_id"] == f176_id]
f176_in_eval_by_frame = [r for r in eval_data if r["anchor_frame"] == f176_frame]

f176_in_supp_by_id = [r for r in supp_data if r["episode_id"] == f176_id]
f176_in_supp_by_frame = [r for r in supp_data if r["anchor_frame"] == f176_frame]

print(f"In evaluation artifact:")
print(f"  By episode_id ('{f176_id}'): {len(f176_in_eval_by_id)} match(es)")
if f176_in_eval_by_id:
    print(f"    Record: frame={f176_in_eval_by_id[0]['anchor_frame']}, score={f176_in_eval_by_id[0]['score']}, disp={f176_in_eval_by_id[0]['disposition']}")
print(f"  By anchor_frame ({f176_frame}): {len(f176_in_eval_by_frame)} match(es)")

print(f"\nIn suppression artifact:")
print(f"  By episode_id ('{f176_id}'): {len(f176_in_supp_by_id)} match(es)")
print(f"  By anchor_frame ({f176_frame}): {len(f176_in_supp_by_frame)} match(es)")

# 5. Remaining evaluation records after excluding confirmed overlaps
# Confirmed overlaps to exclude:
# - F176 (5uh8-3q51uI_00_ep01_f24)
# - The quarantined episodes (intersection with supp_data)
excluded_ids = set([f176_id]).union(intersection)
print(f"\n=== 5. EXCLUSION & REMAINING EVALUATION RECORDS ===")
print(f"Total IDs marked for exclusion: {len(excluded_ids)} (1 F176 + {len(intersection)} quarantined)")
remaining_records = [r for r in eval_data if r["episode_id"] not in excluded_ids]
print(f"Remaining evaluation records count: {len(remaining_records)}")
print(f"Remaining unique episode IDs: {len(set(r['episode_id'] for r in remaining_records))}")

# Check any duplicates in remaining records
from collections import Counter
rem_id_counts = Counter(r["episode_id"] for r in remaining_records)
rem_dup_ids = {k: v for k, v in rem_id_counts.items() if v > 1}
print(f"Duplicate episode IDs in remaining records: {rem_dup_ids}")

rem_frame_counts = Counter(r["anchor_frame"] for r in remaining_records)
rem_dup_frames = {k: v for k, v in rem_frame_counts.items() if v > 1}
print(f"Duplicate anchor frames in remaining records: {rem_dup_frames}")
for f_dup in rem_dup_frames:
    matches = [r for r in remaining_records if r["anchor_frame"] == f_dup]
    print(f"  Frame {f_dup}: {[m['episode_id'] for m in matches]}")
