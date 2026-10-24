import json

with open('artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json', 'r') as f:
    cohort = json.load(f)

print(f"Total cohort size: {len(cohort)}")
for i, ep in enumerate(cohort):
    up = ep.get('telemetry', {}).get('upstream_react', None)
    ctype = ep.get('canonical_type', '')
    disp = ep.get('disposition', '')
    score = ep.get('score', 0)
    epid = ep.get('episode_id', '')
    anchor = ep.get('anchor_frame', '')
    print(f"{i:02d}: ep={epid:<25} anchor={anchor} type={ctype:<7} disp={disp:<30} score={score:.4f} up={up}")
