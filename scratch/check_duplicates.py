import json
from collections import Counter

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)

# Check all anchor frames in the 53 records
frames = [ep["anchor_frame"] for ep in eval_audit]
counts = Counter(frames)
duplicates_all = {f: c for f, c in counts.items() if c > 1}
print(f"All duplicate anchor frames in the 53-record file: {duplicates_all}")

for f in duplicates_all:
    matching = [ep for ep in eval_audit if ep["anchor_frame"] == f]
    print(f"\nFrame {f} appears {len(matching)} times:")
    for m in matching:
        print(f"  ep_id={m['episode_id']}, disp={m['disposition']}, score={m['score']}")

# Now verify the exact 36 remaining
f176_id = "5uh8-3q51uI_00_ep01_f24"
remaining_36 = [ep for ep in eval_audit if ep["episode_id"] != f176_id and ep["episode_id"] not in supp_ep_ids]

print(f"\nRemaining count: {len(remaining_36)}")

# Check duplicate anchor frames within the remaining 36
rem_frames = [ep["anchor_frame"] for ep in remaining_36]
rem_counts = Counter(rem_frames)
rem_duplicates = {f: c for f, c in rem_counts.items() if c > 1}
print(f"Duplicate anchor frames within the remaining 36: {rem_duplicates}")
for f in rem_duplicates:
    matching = [ep for ep in remaining_36 if ep["anchor_frame"] == f]
    print(f"  Frame {f}: {[m['episode_id'] for m in matching]}")
