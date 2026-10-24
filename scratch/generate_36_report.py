import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)
f176_id = "5uh8-3q51uI_00_ep01_f24"

remaining_36 = [ep for ep in eval_audit if ep["episode_id"] != f176_id and ep["episode_id"] not in supp_ep_ids]

print(f"Total remaining records: {len(remaining_36)}")

channels_list = ["phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat", "phi_push", "phi_up"]

rows = []
for ep in remaining_36:
    eid = ep["episode_id"]
    clip_id = "5uh8-3q51uI_00"
    anchor_frame = ep["anchor_frame"]
    canon_type = ep["canonical_type"]
    source_run = "aftermath_evaluation_audit_5uh8-3q51uI_00.json"
    score = ep.get("score")
    disp = ep.get("disposition")
    disp_prov = "AUTOMATED_RULE_OUTPUT"
    
    ch = ep.get("channels", {})
    telem = ep.get("telemetry", {})
    
    missing = []
    if score is None: missing.append("baseline_score")
    for c in channels_list:
        if ch.get(c) is None: missing.append(c)
    if telem.get("v_entry") is None: missing.append("v_entry")
    if telem.get("delta_theta") is None: missing.append("delta_theta")
    
    missing_str = "NONE" if not missing else ", ".join(missing)
    
    row = {
        "episode_id": eid,
        "clip_id": clip_id,
        "anchor_frame": anchor_frame,
        "canonical_type": canon_type,
        "source_run_id": source_run,
        "baseline_score": f"{score:.4f}" if score is not None else "MISSING",
        "phi_dv": f"{ch['phi_dv']:.4f}" if "phi_dv" in ch else "MISSING",
        "phi_rest": f"{ch['phi_rest']:.4f}" if "phi_rest" in ch else "MISSING",
        "phi_rot": f"{ch['phi_rot']:.4f}" if "phi_rot" in ch else "MISSING",
        "phi_ovl": f"{ch['phi_ovl']:.4f}" if "phi_ovl" in ch else "MISSING",
        "phi_lat": f"{ch['phi_lat']:.4f}" if "phi_lat" in ch else "MISSING",
        "phi_push": f"{ch['phi_push']:.4f}" if "phi_push" in ch else "MISSING",
        "phi_up": f"{ch['phi_up']:.4f}" if "phi_up" in ch else "MISSING",
        "v_entry": f"{telem['v_entry']:.2f}" if "v_entry" in telem else "MISSING",
        "delta_theta": f"{telem['delta_theta']:.2f}" if "delta_theta" in telem else "MISSING",
        "disposition": disp,
        "disposition_provenance": disp_prov,
        "missing_fields": missing_str
    }
    rows.append(row)

with open("scratch/extracted_36_rows.json", "w", encoding="utf-8") as out:
    json.dump(rows, out, indent=2)

print("Saved to scratch/extracted_36_rows.json")

# Print first 5 and last 5
print("\nFirst 3 rows:")
for r in rows[:3]:
    print(r)
print("\nLast 2 rows:")
for r in rows[-2:]:
    print(r)
