"""Forensic inspection of the 10 surviving Path A/B triggers on 5uh8-3q51uI_00.
Strictly READ-ONLY: extracts track histories, bounding boxes, ground kinematics,
and temporal evolution across [t - 30, t + 30].
"""

import os
import sys
import json
import numpy as np

# Load audit data
with open("artifacts/stage_t_path_c_v3_audit.json") as f:
    audit = json.load(f)

cands = [c for c in audit["all_candidates"] if c["event_type"] == "distant_trajectory_conflict"]
print(f"Total distant conflicts: {len(cands)}")
for c in cands:
    sig = c.get("signals", {})
    print(f"F{c['frame']} tracks={c['track_ids']} domain={sig.get('conflict_domain')} "
          f"d_lat={sig.get('d_lat',0):.2f} dist={sig.get('distance_m',0):.2f} "
          f"v_rel={sig.get('relative_speed_kmh',0):.1f} desc={c.get('description')}")
