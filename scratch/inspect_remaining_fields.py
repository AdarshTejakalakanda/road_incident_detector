import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)
f176_id = "5uh8-3q51uI_00_ep01_f24"

remaining_36 = [ep for ep in eval_audit if ep["episode_id"] != f176_id and ep["episode_id"] not in supp_ep_ids]

print("First record of remaining 36:")
print(json.dumps(remaining_36[0], indent=2))

# Check fields across all 36 records
all_keys = set()
for ep in remaining_36:
    all_keys.update(ep.keys())
print("\nAll top-level keys across remaining 36:", sorted(list(all_keys)))

# Check channels and telemetry sub-keys
ch_keys = set()
telem_keys = set()
for ep in remaining_36:
    if "channels" in ep and isinstance(ep["channels"], dict):
        ch_keys.update(ep["channels"].keys())
    if "telemetry" in ep and isinstance(ep["telemetry"], dict):
        telem_keys.update(ep["telemetry"].keys())

print("Channels keys:", sorted(list(ch_keys)))
print("Telemetry keys:", sorted(list(telem_keys)))
