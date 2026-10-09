import json

with open("artifacts/stage_t_physical_gate_audit.json") as f:
    d = json.load(f)

cands = [c for c in d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
path_c_cands = [c for c in cands if c.get("signals", {}).get("conflict_domain") == "DOMAIN_2_PATH_C"]

print(f"Total Path C candidates: {len(path_c_cands)}")
for c in path_c_cands[:10]:
    sig = c.get("signals", {})
    print(f"F{c['frame']} tracks={c['track_ids']}: dist={sig.get('distance_m',0):.2f}m, delta_theta={sig.get('delta_theta_deg',0):.2f}deg")
