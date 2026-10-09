import json

with open("artifacts/stage_t_physical_gate_audit.json") as f:
    d = json.load(f)

cands = [c for c in d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict" and c.get("signals", {}).get("conflict_domain") == "DOMAIN_2_PATH_C"]

print(f"Total Path C: {len(cands)}")
for i, c in enumerate(cands):
    sig = c.get("signals", {})
    print(f"#{i+1}: Frame {c['frame']} tracks={c['track_ids']} dist={sig.get('distance_m', 0.0):.2f}m")
