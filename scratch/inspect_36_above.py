import json

with open('artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json') as f:
    eval_audit = json.load(f)
with open('artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json') as f:
    supp_audit = json.load(f)

supp_ids = set(e['episode_id'] for e in supp_audit)
f176_id = '5uh8-3q51uI_00_ep01_f24'
remaining_36 = [ep for ep in eval_audit if ep['episode_id'] != f176_id and ep['episode_id'] not in supp_ids]

print(f"Total remaining: {len(remaining_36)}")
for ep in remaining_36:
    print(f"{ep['episode_id']}: type={ep['canonical_type']}, frame={ep['anchor_frame']}, score={ep['score']}, up={ep['telemetry'].get('upstream_react')}")
