"""Map the exact suppression mechanism for each of the 19 prior Path-C frames."""

import json

with open("scratch/path_c_forensic_results.json") as f:
    forensics = json.load(f)

print(f"Loaded {len(forensics)} forensic records.")
for rec in forensics:
    rec_id = rec["rec_id"]
    f = rec["frame"]
    tracks = rec["tracks"]
    trig_t = rec["trigger_track"]
    dtheta_5f = rec["delta_theta_5f"]
    a_lat = rec["a_lat"]
    planar_dist = rec["planar_dist"]

    # Rejection mechanism analysis:
    # Condition 1: dist <= 8.0 m, d_lat < 1.80 m, ground_history >= 6
    # Condition 2: delta_theta_5f >= 15.0 deg
    # Condition 3: a_lat <= 10.0 m/s^2
    reasons = []
    if dtheta_5f < 15.0:
        reasons.append(f"Sub-threshold net displacement (delta_theta_5f={dtheta_5f:.2f} deg < 15.0 deg)")
    if a_lat > 10.0:
        reasons.append(f"Non-physical coordinate flip / snap (a_lat={a_lat:.2f} m/s^2 > 10.0 m/s^2)")

    status = "REJECTED: " + " AND ".join(reasons) if reasons else "ADMITTED"
    print(f"Record #{rec_id} (Frame {f}, Tracks {tracks}): {status}")
