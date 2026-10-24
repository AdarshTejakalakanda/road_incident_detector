import json

with open("scratch/extracted_36_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

# Task 1: Print list of (episode_id, anchor_frame)
print("=== TASK 1 ENUMERATION ===")
for i, r in enumerate(rows):
    print(f"{i+1:2d}. {r['episode_id']} (anchor frame {r['anchor_frame']})")

# Task 2: Print Markdown table
print("\n=== TASK 2 FULL TABLE ===")
header = "| episode_id | clip_id | anchor_frame | canonical_type | source_run_id | baseline_score | phi_dv | phi_rest | phi_rot | phi_ovl | phi_lat | phi_push | phi_up | v_entry | delta_theta | disposition | disposition_provenance | missing_fields |"
sep = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
print(header)
print(sep)
for r in rows:
    line = f"| {r['episode_id']} | {r['clip_id']} | {r['anchor_frame']} | {r['canonical_type']} | {r['source_run_id']} | {r['baseline_score']} | {r['phi_dv']} | {r['phi_rest']} | {r['phi_rot']} | {r['phi_ovl']} | {r['phi_lat']} | {r['phi_push']} | {r['phi_up']} | {r['v_entry']} | {r['delta_theta']} | {r['disposition']} | {r['disposition_provenance']} | {r['missing_fields']} |"
    print(line)

# Task 3: Field coverage matrix (rows = episodes, columns = required fields)
fields = [
    "episode_id", "clip_id", "anchor_frame", "canonical_type", "source_run_id",
    "baseline_score", "phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat",
    "phi_push", "phi_up", "v_entry", "delta_theta", "disposition",
    "disposition_provenance", "missing_fields"
]

print("\n=== TASK 3 FIELD COVERAGE MATRIX (AGGREGATE COUNTS) ===")
for f in fields:
    present_cnt = sum(1 for r in rows if r[f] != "MISSING")
    missing_cnt = 36 - present_cnt
    print(f"{f}: {present_cnt}/36 PRESENT ({missing_cnt} MISSING)")
