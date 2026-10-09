"""Investigate Record #18 (F835) in Path C v3 run."""

import json

with open("artifacts/stage_t_path_c_v3_audit.json") as f:
    d = json.load(f)

cands = [c for c in d["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
f835 = [c for c in cands if c["frame"] == 835]
print("F835 candidates in Path C v3 run:", f835)
