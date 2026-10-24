import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)
f176_id = "5uh8-3q51uI_00_ep01_f24"

remaining_36 = [ep for ep in eval_audit if ep["episode_id"] != f176_id and ep["episode_id"] not in supp_ep_ids]

print(f"Count of remaining records: {len(remaining_36)}")

disps = {}
for ep in remaining_36:
    d = ep.get("disposition")
    disps[d] = disps.get(d, 0) + 1
print(f"Dispositions in remaining 36: {disps}")

min_score = min(ep["score"] for ep in remaining_36)
max_score = max(ep["score"] for ep in remaining_36)
print(f"Score range in remaining 36: min={min_score}, max={max_score}")

# Check if any field is missing
required_fields = ["episode_id", "clip_id", "anchor_frame", "canonical_type", "source_run_id",
                   "baseline_score", "phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat",
                   "phi_push", "phi_up", "v_entry", "delta_theta", "disposition",
                   "disposition_provenance", "missing_fields"]

for ep in remaining_36:
    eid = ep.get("episode_id")
    ch = ep.get("channels", {})
    telem = ep.get("telemetry", {})
    missing = []
    if ep.get("anchor_frame") is None: missing.append("anchor_frame")
    if ep.get("canonical_type") is None: missing.append("canonical_type")
    if ep.get("score") is None: missing.append("baseline_score")
    for c in ["phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat", "phi_push", "phi_up"]:
        if ch.get(c) is None: missing.append(c)
    if telem.get("v_entry") is None: missing.append("v_entry")
    if telem.get("delta_theta") is None: missing.append("delta_theta")
    if ep.get("disposition") is None: missing.append("disposition")
    if missing:
        print(f"Episode {eid} missing: {missing}")

print("Check complete. All 36 checked for missing values.")
