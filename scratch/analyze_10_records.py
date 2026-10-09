"""Analyze the 10 records and crash sequence from forensic_10_records_data.json."""

import json

with open("scratch/forensic_10_records_data.json") as f:
    data = json.load(f)

records = data["records"]
crash_seq = data["crash_sequence"]

print("Loaded records:", len(records))

for rid_str, rdata in records.items():
    rinfo = rdata["record"]
    f = rinfo["frame"]
    t1, t2 = rinfo["tracks"]
    path = rinfo["path"]
    t1_hist = rdata["t1_hist"]
    t2_hist = rdata["t2_hist"]
    
    print(f"\n=======================================================")
    print(f"Record #{rid_str}: Frame {f}, Tracks [{t1}, {t2}], Path {path}")
    print(f"Track {t1} present in {len(t1_hist)} frames, Track {t2} in {len(t2_hist)} frames")
    
    # State at trigger frame
    s1 = t1_hist.get(str(f))
    s2 = t2_hist.get(str(f))
    if s1 and s2:
        print(f"  At F{f}:")
        print(f"    T{t1}: pos={s1['ground_pos']}, spd={s1['speed_kmh']}km/h, acc={s1['accel_ms2']}m/s2, bbox={s1['bbox']}, cls={s1['class_label']}")
        print(f"    T{t2}: pos={s2['ground_pos']}, spd={s2['speed_kmh']}km/h, acc={s2['accel_ms2']}m/s2, bbox={s2['bbox']}, cls={s2['class_label']}")
        d_lat = abs(s1['ground_pos'][0] - s2['ground_pos'][0])
        d_long = abs(s1['ground_pos'][1] - s2['ground_pos'][1])
        print(f"    d_lat={d_lat:.2f}m, d_long={d_long:.2f}m")
    
    # State 15 frames before
    f_pre = str(f - 15)
    s1_pre = t1_hist.get(f_pre)
    s2_pre = t2_hist.get(f_pre)
    if s1_pre and s2_pre:
        print(f"  At F{f-15} (pre):")
        print(f"    T{t1}: spd={s1_pre['speed_kmh']}km/h, pos={s1_pre['ground_pos']}")
        print(f"    T{t2}: spd={s2_pre['speed_kmh']}km/h, pos={s2_pre['ground_pos']}")

    # State 15 frames after
    f_post = str(f + 15)
    s1_post = t1_hist.get(f_post)
    s2_post = t2_hist.get(f_post)
    if s1_post and s2_post:
        print(f"  At F{f+15} (post):")
        print(f"    T{t1}: spd={s1_post['speed_kmh']}km/h, pos={s1_post['ground_pos']}")
        print(f"    T{t2}: spd={s2_post['speed_kmh']}km/h, pos={s2_post['ground_pos']}")
