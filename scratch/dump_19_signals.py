import json

with open("artifacts/stage_t_two_domain_patched_audit.json") as f:
    d = json.load(f)

evs = [c for c in d.get("all_candidates", []) if c.get("event_type") == "distant_trajectory_conflict"]
print(f"Total distant conflicts: {len(evs)}")
for i, e in enumerate(evs):
    sig = e.get("signals", {})
    print(f"#{i+1}: Frame {e['frame']} tracks={e['track_ids']}: "
          f"d_lat={sig.get('d_lat', 0.0):.2f}m, "
          f"delta_d_lat={sig.get('delta_d_lat', 0.0):.2f}m, "
          f"dist={sig.get('distance_m', 0.0):.1f}m, "
          f"v_rel={sig.get('relative_speed_kmh', 0.0):.1f}km/h, "
          f"dca_5f={sig.get('dca_m', 0.0):.2f}m, "
          f"tca_5f={sig.get('tca_sec', 0.0):.2f}s, "
          f"acc1={sig.get('metric_accel_1', 0.0):.2f}, "
          f"acc2={sig.get('metric_accel_2', 0.0):.2f}")
