import json

with open("artifacts/stage_t_remediated_5uh8_audit.json") as f:
    audit = json.load(f)

colls = [c for c in audit["all_candidates"] if c.get("event_type") == "collision_candidate" and c.get("frame") != 182]

for i in range(14):
    c = colls[i]
    print(f"FP-{i+1:02d}: frame={c.get('frame')}, tracks={c.get('track_ids')}")
