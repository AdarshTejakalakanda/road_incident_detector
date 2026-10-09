"""Check why Record #18 was not emitted in distant_conflict detector."""

import os, sys, json
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Let's inspect the suppressed events log from the detector or run a mini check on F835
with open("artifacts/stage_t_path_c_v3_audit.json") as f:
    d = json.load(f)

# Let's see if F835 pair [229, 246] was evaluated
cands = [c for c in d["all_candidates"] if 830 <= c["frame"] <= 840]
print("All candidates near 835:")
for c in cands:
    print(c["frame"], c["event_type"], c["track_ids"])
