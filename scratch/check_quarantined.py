import json

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp = json.load(f)

print(f"Quarantined count: {len(supp)}")
for item in supp:
    print(f"id={item.get('episode_id')}, frame={item.get('anchor_frame')}, type={item.get('canonical_type')}, score={item.get('composite_score')}")
