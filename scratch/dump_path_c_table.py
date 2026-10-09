import json

with open("scratch/path_c_forensic_results.json") as f:
    data = json.load(f)

for d in data:
    print(f"#{d['rec_id']}: F{d['frame']} tracks={d['tracks']} trig_track={d['trigger_track']} "
          f"dist={d['planar_dist']:.2f}m dtheta_inst={d['delta_theta_inst']:.2f}deg "
          f"dtheta_3f={d['delta_theta_3f']:.2f}deg dtheta_5f={d['delta_theta_5f']:.2f}deg dtheta_10f={d['delta_theta_10f']:.2f}deg "
          f"N_persist={d['n_persist']} a_lat={d['a_lat']:.2f}m/s2 v={d['v_mps']*3.6:.1f}km/h")
