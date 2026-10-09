import json

with open("scratch/extracted_42_rows.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

print(f"Total rows: {len(rows)}")

# Task 3: Ground-Truth Audit
dispositions = {}
for r in rows:
    disp = r["disposition"]
    prov = r["disposition_provenance"]
    key = (disp, prov)
    dispositions[key] = dispositions.get(key, 0) + 1

print("\n--- Distinct Dispositions & Counts ---")
for (d, p), count in dispositions.items():
    print(f"Disposition: {d} | Provenance: {p} | Count: {count}")

# Task 4: Field Coverage Matrix check
required_fields = [
    "episode_id", "clip_id", "anchor_frame", "canonical_type", "source_run_id",
    "baseline_score", "phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat",
    "phi_push", "phi_up", "v_entry", "delta_theta", "disposition",
    "disposition_provenance", "missing_fields"
]

missing_counts = {f: 0 for f in required_fields}
for r in rows:
    for f in required_fields:
        if r.get(f) is None or r.get(f) == "MISSING":
            missing_counts[f] += 1

print("\n--- Field Coverage Check across 42 rows ---")
for f, m in missing_counts.items():
    print(f"Field {f}: {42 - m}/42 PRESENT ({m} MISSING)")
