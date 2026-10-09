import json

with open("scratch/extracted_42_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

header = "| episode_id | clip_id | anchor_frame | canonical_type | source_run_id | baseline_score | phi_dv | phi_rest | phi_rot | phi_ovl | phi_lat | phi_push | phi_up | v_entry | delta_theta | disposition | disposition_provenance | missing_fields |"
sep = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

print(header)
print(sep)
for r in rows:
    line = f"| {r['episode_id']} | {r['clip_id']} | {r['anchor_frame']} | {r['canonical_type']} | {r['source_run_id']} | {r['baseline_score']:.4f} | {r['phi_dv']:.4f} | {r['phi_rest']:.4f} | {r['phi_rot']:.4f} | {r['phi_ovl']:.4f} | {r['phi_lat']:.4f} | {r['phi_push']:.4f} | {r['phi_up']:.4f} | {r['v_entry']:.1f} | {r['delta_theta']:.1f} | {r['disposition']} | {r['disposition_provenance']} | {r['missing_fields']} |"
    print(line)
