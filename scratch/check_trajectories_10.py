"""Check visual crops and scene response for all 10 records."""

import json

with open("scratch/forensic_10_records_data.json") as f:
    data = json.load(f)

records = data["records"]

for rid, rdata in records.items():
    r = rdata["record"]
    f = r["frame"]
    t1, t2 = r["tracks"]
    path = r["path"]
    t1_h = rdata["t1_hist"]
    t2_h = rdata["t2_hist"]
    
    # Analyze speeds and positions
    speeds_1 = [v["speed_kmh"] for v in t1_h.values()]
    speeds_2 = [v["speed_kmh"] for v in t2_h.values()]
    
    pos_1 = [v["ground_pos"] for v in t1_h.values()]
    pos_2 = [v["ground_pos"] for v in t2_h.values()]
    
    print(f"\n--- Record #{rid} (Frame {f}, Tracks [{t1}, {t2}], Path {path}) ---")
    print(f"Track {t1}: Speed range [{min(speeds_1):.1f}, {max(speeds_1):.1f}] km/h, mean={sum(speeds_1)/len(speeds_1):.1f} km/h")
    print(f"Track {t2}: Speed range [{min(speeds_2):.1f}, {max(speeds_2):.1f}] km/h, mean={sum(speeds_2)/len(speeds_2):.1f} km/h")
    print(f"Track {t1} start pos {pos_1[0]} -> end pos {pos_1[-1]}")
    print(f"Track {t2} start pos {pos_2[0]} -> end pos {pos_2[-1]}")
