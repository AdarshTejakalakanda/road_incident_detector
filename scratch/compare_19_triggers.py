import json

with open("artifacts/stage_t_two_domain_patched_audit.json") as f:
    old_d = json.load(f)
with open("artifacts/stage_t_physical_gate_audit.json") as f:
    new_d = json.load(f)

old_cands = [c for c in old_d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
new_cands = [c for c in new_d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]

old_keys = set((c["frame"], tuple(c["track_ids"])) for c in old_cands)
new_keys = set((c["frame"], tuple(c["track_ids"])) for c in new_cands)

survived = old_keys.intersection(new_keys)
print(f"Old count: {len(old_keys)}")
print(f"New count: {len(new_keys)}")
print(f"Surviving from old 19: {len(survived)} -> {survived}")

print("\nDetailed comparison for the 19 old triggers:")
for i, c in enumerate(old_cands):
    k = (c["frame"], tuple(c["track_ids"]))
    passed = k in new_keys
    print(f"Old #{i+1}: Frame {c['frame']} tracks={c['track_ids']} -> {'SURVIVED' if passed else 'SUPPRESSED'}")
