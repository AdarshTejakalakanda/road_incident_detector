import json

with open("scratch/extracted_42_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

for r in rows[:14]:
    line = f"| {r['episode_id']} | {r['clip_id']} | {r['anchor_frame']} | {r['canonical_type']} | {r['source_run_id']} | {r['baseline_score']:.4f} | {r['phi_dv']:.4f} | {r['phi_rest']:.4f} | {r['phi_rot']:.4f} | {r['phi_ovl']:.4f} | {r['phi_lat']:.4f} | {r['phi_push']:.4f} | {r['phi_up']:.4f} | {r['v_entry']:.1f} | {r['delta_theta']:.1f} | {r['disposition']} | {r['disposition_provenance']} | {r['missing_fields']} |"
    print(line)
