"""Inspect track trajectory details and visual behavior for specific records:
Rec #3 (F197, tracks [49, 77]) - near crash zone
Rec #12 (F713, track 132) & Rec #13 (F716, track 132)
Rec #1 (F21, tracks [4, 5])
Rec #5 (F376, tracks [100, 109])
Rec #9 (F512, tracks [163, 169])
Rec #15 (F728, tracks [157, 213])
"""

import json

with open("scratch/path_c_forensic_results.json") as f:
    data = json.load(f)

for d in data:
    f = d["frame"]
    t1, t2 = d["tracks"]
    trig_t = d["trigger_track"]
    print(f"--- Record #{d['rec_id']}: Frame {f}, Tracks {[t1, t2]}, Triggered by Track {trig_t} ---")
    print(f"Planar Dist: {d['planar_dist']:.2f}m")
    print(f"Heading Changes: Inst={d['delta_theta_inst']:.2f}deg, 3f={d['delta_theta_3f']:.2f}deg, 5f={d['delta_theta_5f']:.2f}deg, 10f={d['delta_theta_10f']:.2f}deg")
    print(f"Yaw Rate Inst: {d['yaw_rate_inst']:.1f} deg/s, a_lat: {d['a_lat']:.2f} m/s^2, speed: {d['v_mps']*3.6:.1f} km/h")
    print(f"N_persist: {d['n_persist']} consecutive frames >= 15 deg/s")
