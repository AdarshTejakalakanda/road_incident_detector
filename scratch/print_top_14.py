import json

with open("scratch/extracted_36_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

for r in rows[:14]:
    line = f"| {r['episode_id']} | {r['clip_id']} | {r['anchor_frame']} | {r['canonical_type']} | {r['source_run_id']} | {r['baseline_score']} | {r['phi_dv']} | {r['phi_rest']} | {r['phi_rot']} | {r['phi_ovl']} | {r['phi_lat']} | {r['phi_push']} | {r['phi_up']} | {r['v_entry']} | {r['delta_theta']} | {r['disposition']} | {r['disposition_provenance']} | {r['missing_fields']} |"
    print(line)
