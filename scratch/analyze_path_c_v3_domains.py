import json

with open("artifacts/stage_t_path_c_v3_audit.json") as f:
    d = json.load(f)

cands = [c for c in d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
print(f"Total emitted distant conflicts: {len(cands)}")
domain_counts = {}
for i, c in enumerate(cands):
    sig = c.get("signals", {})
    cdom = sig.get("conflict_domain", "UNKNOWN")
    domain_counts[cdom] = domain_counts.get(cdom, 0) + 1
    print(f"#{i+1}: F{c['frame']} tracks={c['track_ids']}: domain={cdom}, d_lat={sig.get('d_lat',0):.2f}m, dist={sig.get('distance_m',0):.1f}m, v_rel={sig.get('relative_speed_kmh',0):.1f}km/h")

print("\nDomain breakdown:")
for k, v in domain_counts.items():
    print(f"  {k}: {v}")
