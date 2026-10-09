"""Examine video frames and bounding boxes around the 19 Path C frames.
Extract bounding box centers, coordinates, and physical scene dynamics across [t-30, t+30].
"""

import json
import numpy as np

with open("scratch/path_c_forensic_results.json") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records.")
for d in data:
    rec_id = d["rec_id"]
    f = d["frame"]
    tracks = d["tracks"]
    trig_t = d["trigger_track"]
    print(f"Rec #{rec_id} (F{f}, Tracks {tracks}, Trigger Track {trig_t}):")
    print(f"  Planar dist: {d['planar_dist']:.2f}m")
    print(f"  dtheta_inst: {d['delta_theta_inst']:.2f} deg, yaw_rate: {d['yaw_rate_inst']:.1f} deg/s")
    print(f"  dtheta_3f: {d['delta_theta_3f']:.2f} deg, dtheta_5f: {d['delta_theta_5f']:.2f} deg, dtheta_10f: {d['delta_theta_10f']:.2f} deg")
    print(f"  N_persist: {d['n_persist']} frames, a_lat: {d['a_lat']:.2f} m/s^2, speed: {d['v_mps']*3.6:.1f} km/h")
