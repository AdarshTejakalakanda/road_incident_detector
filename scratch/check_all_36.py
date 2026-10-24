import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)
f176_id = "5uh8-3q51uI_00_ep01_f24"

remaining_36 = [ep for ep in eval_audit if ep["episode_id"] != f176_id and ep["episode_id"] not in supp_ep_ids]

channels_list = ["phi_dv", "phi_rest", "phi_rot", "phi_ovl", "phi_lat", "phi_push", "phi_up"]

for i, ep in enumerate(remaining_36):
    eid = ep.get("episode_id")
    score = ep.get("score")
    ch = ep.get("channels", {})
    telem = ep.get("telemetry", {})
    missing = []
    if score is None: missing.append("baseline_score")
    for c in channels_list:
        if ch.get(c) is None: missing.append(c)
    if telem.get("v_entry") is None: missing.append("v_entry")
    if telem.get("delta_theta") is None: missing.append("delta_theta")
    if missing:
        print(f"Row {i+1} ({eid}, F{ep.get('anchor_frame')}): MISSING -> {missing}")
    else:
        # Check types and ranges
        assert isinstance(score, (int, float))
        for c in channels_list:
            assert isinstance(ch[c], (int, float))
        assert isinstance(telem["v_entry"], (int, float))
        assert isinstance(telem["delta_theta"], (int, float))

print(f"Inspection finished across all {len(remaining_36)} records.")
