import json

with open("artifacts/stage_t_physical_gate_audit.json") as f:
    d = json.load(f)

cands = [c for c in d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
print(f"Total distant conflicts emitted: {len(cands)}")
for i, c in enumerate(cands):
    sig = c.get("signals", {})
    print(f"#{i+1}: Frame {c['frame']} tracks={c['track_ids']}: "
          f"d_lat={sig.get('d_lat', 0.0):.2f}m, "
          f"delta_d_lat={sig.get('delta_d_lat', 0.0):.2f}m, "
          f"dist={sig.get('distance_m', 0.0):.1f}m, "
          f"v_rel={sig.get('relative_speed_kmh', 0.0):.1f}km/h, "
          f"dca_5f={sig.get('dca_m', 0.0):.2f}m, "
          f"tca_5f={sig.get('tca_sec', 0.0):.2f}s, "
          f"desc={c.get('description')}")
