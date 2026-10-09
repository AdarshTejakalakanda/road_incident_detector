import json

with open("artifacts/stage_t_two_domain_patched_audit.json") as f:
    d = json.load(f)

print("In-window episodes data:")
for ep in d.get("in_window_episodes_data", []):
    print(ep)

print("\nAll candidates in F176 +/- 15:")
for c in d.get("all_candidates", []):
    if abs(c["frame"] - 176) <= 15:
        print(f"F{c['frame']} {c['event_type']} tracks={c['track_ids']} conf={c['confidence']} sigs={c.get('signals')}")
