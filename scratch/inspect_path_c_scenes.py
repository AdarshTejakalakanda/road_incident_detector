"""Examine physical motion for all 19 records.
Specifically, inspect:
1. Is the vehicle driving straight down the highway lane or undergoing true evasive steering?
2. What are the track bounding boxes across [t-15, t+15]?
3. Does the vehicle cross lane markings or swerve away from another vehicle?
"""

import os
import sys
import json
import cv2
import numpy as np

with open("scratch/path_c_forensic_results.json") as f:
    records = json.load(f)

# Also load all candidates from the audit JSON to inspect signals
with open("artifacts/stage_t_physical_gate_audit.json") as f:
    audit = json.load(f)

path_c_audit = [c for c in audit["all_candidates"] if c["event_type"] == "distant_trajectory_conflict" and c.get("signals", {}).get("conflict_domain") == "DOMAIN_2_PATH_C"]

print(f"Loaded {len(path_c_audit)} audit records matching Path C.")

for rec in records:
    r_id = rec["rec_id"]
    f = rec["frame"]
    t1, t2 = rec["tracks"]
    trig_t = rec["trigger_track"]
    print(f"\n=======================================================")
    print(f"Record #{r_id}: Frame {f}, Tracks {t1} and {t2} (Trigger Track {trig_t})")
    print(f"Planar Dist: {rec['planar_dist']:.2f}m")
    print(f"Inst Yaw Rate: {rec['yaw_rate_inst']:.1f} deg/s (dtheta_inst={rec['delta_theta_inst']:.2f} deg)")
    print(f"Multi-frame dtheta: 3f={rec['delta_theta_3f']:.2f} deg, 5f={rec['delta_theta_5f']:.2f} deg, 10f={rec['delta_theta_10f']:.2f} deg")
    print(f"N_persist: {rec['n_persist']} frames, a_lat: {rec['a_lat']:.2f} m/s^2, speed: {rec['v_mps']*3.6:.1f} km/h")
