import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle

with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    vids = pickle.load(f)
v = [x for x in vids if x["video_id"] == "kABWTDKgc9U_00"][0]

for f_idx, pts, tr, kin, conv in v["frames"]:
    if 195 <= f_idx <= 205:
        t27, t28 = tr.get(27), tr.get(28)
        k27, k28 = kin.get(27), kin.get(28)
        c_pair = [c for c in conv if (c.track_id_1, c.track_id_2) in [(27, 28), (28, 27)]]
        print(f"f={f_idx}: t27={'yes' if t27 else 'no'} (age={t27.track_age if t27 else 0}, b={t27.current_bbox_xywh if t27 else 0}), t28={'yes' if t28 else 'no'} (age={t28.track_age if t28 else 0}, b={t28.current_bbox_xywh if t28 else 0})")
        if c_pair:
            c = c_pair[0]
            print(f"   conv: dist={c.distance_m:.1f}m, conv={c.is_converging}, tca={c.tca_sec:.2f}s, dca={c.dca_m:.1f}m")
        if k27 and k28:
            print(f"   kin: acc27={k27.acceleration_mag:.1f}, acc28={k28.acceleration_mag:.1f}, spd27={k27.speed_kmh:.1f}, spd28={k28.speed_kmh:.1f}")
