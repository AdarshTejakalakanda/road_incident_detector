import json

path_eval = "artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json"
path_supp = "artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json"

with open(path_eval, "r", encoding="utf-8") as f:
    eval_list = json.load(f)

with open(path_supp, "r", encoding="utf-8") as f:
    supp_list = json.load(f)

# Episode IDs in order
eval_ids = [r["episode_id"] for r in eval_list]
supp_ids = [r["episode_id"] for r in supp_list]

print("=== EVALUATION ARTIFACT (53 RECORDS) ===")
for i, r in enumerate(eval_list):
    print(f"[{i+1:02d}] {r['episode_id']} | frame={r['anchor_frame']} | type={r.get('canonical_type')} | score={r.get('score')} | disp={r.get('disposition')}")

print("\n=== SUPPRESSION ARTIFACT (16 RECORDS) ===")
for i, r in enumerate(supp_list):
    print(f"[{i+1:02d}] {r['episode_id']} | frame={r['anchor_frame']} | score={r.get('score')} | disp={r.get('disposition')}")

# Set operations
s_eval = set(eval_ids)
s_supp = set(supp_ids)

inter = s_eval.intersection(s_supp)
diff_supp_eval = s_supp - s_eval
diff_eval_supp = s_eval - s_supp

print(f"\nIntersection count: {len(inter)}")
print(f"Supp - Eval count: {len(diff_supp_eval)}")
print(f"Eval - Supp count: {len(diff_eval_supp)}")

# F176
f176_id = "5uh8-3q51uI_00_ep01_f24"
print(f"\nF176 in eval: {f176_id in s_eval}")
print(f"F176 in supp: {f176_id in s_supp}")

# Exclude F176 + intersection
to_exclude = set([f176_id]).union(inter)
print(f"\nTotal excluded: {len(to_exclude)}")

remaining = [r for r in eval_list if r["episode_id"] not in to_exclude]
print(f"Remaining evaluation records count: {len(remaining)}")

print("\n=== REMAINING EVALUATION RECORDS (EXACT COUNT: %d) ===" % len(remaining))
for i, r in enumerate(remaining):
    print(f"[{i+1:02d}] {r['episode_id']} | frame={r['anchor_frame']} | type={r.get('canonical_type')} | score={r.get('score')} | disp={r.get('disposition')}")
