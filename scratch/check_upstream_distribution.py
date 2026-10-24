import json

with open('artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json') as f:
    cohort = json.load(f)

upstream_counts = []
for ep in cohort:
    telem = ep.get('telemetry', {})
    if 'upstream_react' in telem and telem['upstream_react'] is not None:
        upstream_counts.append(telem['upstream_react'])

print(f"Total episodes with upstream_react: {len(upstream_counts)}")
print(f"Min upstream_react: {min(upstream_counts)}")
print(f"Max upstream_react: {max(upstream_counts)}")
print(f"Distribution: {sorted(upstream_counts)}")
