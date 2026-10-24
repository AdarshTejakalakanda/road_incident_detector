import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

print(f"Total records in aftermath_evaluation_audit_5uh8-3q51uI_00.json: {len(eval_audit)}")

# Check F176
f176 = [ep for ep in eval_audit if ep.get("episode_id") == "5uh8-3q51uI_00_ep01_f24" or ep.get("anchor_frame") == 182]
print(f"F176 matches: {len(f176)}, ids={[e.get('episode_id') for e in f176]}, frames={[e.get('anchor_frame') for e in f176]}")

# 16 quarantined episodes
quarantine_frames = [37, 64, 147, 180, 205, 328, 433, 471, 474, 483, 532, 570, 595, 681, 711, 720]

# Let's load aftermath_suppression_audit_5uh8-3q51uI_00.json to get exact episode_ids of the 16
with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)
supp_ep_ids = set(e["episode_id"] for e in supp_audit)
print(f"Suppressed audit episode count: {len(supp_ep_ids)}")
print(f"Suppressed ep_ids: {sorted(list(supp_ep_ids))}")

# Remaining records
remaining = []
quarantined_matches = []
f176_matches = []

for ep in eval_audit:
    eid = ep.get("episode_id")
    frame = ep.get("anchor_frame")
    if eid == "5uh8-3q51uI_00_ep01_f24" or frame == 182:
        f176_matches.append(ep)
    elif eid in supp_ep_ids or frame in quarantine_frames:
        quarantined_matches.append(ep)
    else:
        remaining.append(ep)

print(f"F176 count: {len(f176_matches)}")
print(f"Quarantined count: {len(quarantined_matches)}")
print(f"Remaining count: {len(remaining)}")

