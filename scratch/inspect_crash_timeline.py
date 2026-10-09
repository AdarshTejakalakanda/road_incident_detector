"""Examine the crash sequence around F176 in forensic_10_records_data.json.
Specifically check tracks 6, 49, 74, 77, and other tracks in F165-F215.
"""

import json

with open("scratch/forensic_10_records_data.json") as f:
    data = json.load(f)

crash_seq = data["crash_sequence"]

print("Crash sequence frames:", min(int(f) for f in crash_seq.keys()), "to", max(int(f) for f in crash_seq.keys()))

for f in range(165, 206, 5):
    f_str = str(f)
    frame_dets = crash_seq.get(f_str, {})
    print(f"\n--- Frame {f} ---")
    for tid, info in frame_dets.items():
        if tid in ("6", "49", "74", "77", "56", "62", "73"):
            print(f"  Track {tid} ({info['class_label']}): pos={info['ground_pos']}, spd={info['speed_kmh']}km/h, bbox={info['bbox']}")

# Also let's trace Track 6 and Track 49 life cycle
t6_frames = []
t49_frames = []
t74_frames = []
t77_frames = []
for f_str, frame_dets in crash_seq.items():
    f = int(f_str)
    if "6" in frame_dets:
        t6_frames.append((f, frame_dets["6"]))
    if "49" in frame_dets:
        t49_frames.append((f, frame_dets["49"]))
    if "74" in frame_dets:
        t74_frames.append((f, frame_dets["74"]))
    if "77" in frame_dets:
        t77_frames.append((f, frame_dets["77"]))

print("\nTrack 6 active range:", min(x[0] for x in t6_frames) if t6_frames else "None", "to", max(x[0] for x in t6_frames) if t6_frames else "None")
print("Track 49 active range:", min(x[0] for x in t49_frames) if t49_frames else "None", "to", max(x[0] for x in t49_frames) if t49_frames else "None")
print("Track 74 active range:", min(x[0] for x in t74_frames) if t74_frames else "None", "to", max(x[0] for x in t74_frames) if t74_frames else "None")
print("Track 77 active range:", min(x[0] for x in t77_frames) if t77_frames else "None", "to", max(x[0] for x in t77_frames) if t77_frames else "None")

# Inspect transition F174 to F185 for T6 and T49
print("\n--- Transition around impact (F174 to F185) ---")
for f in range(174, 186):
    f_str = str(f)
    dets = crash_seq.get(f_str, {})
    t6_info = dets.get("6", {})
    t49_info = dets.get("49", {})
    print(f"F{f}: T6 spd={t6_info.get('speed_kmh')} pos={t6_info.get('ground_pos')} | T49 spd={t49_info.get('speed_kmh')} pos={t49_info.get('ground_pos')}")

# Inspect appearance of T74 and T77
print("\n--- Appearance of T74 and T77 (F185 to F202) ---")
for f in range(185, 203):
    f_str = str(f)
    dets = crash_seq.get(f_str, {})
    t49_info = dets.get("49", {})
    t74_info = dets.get("74", {})
    t77_info = dets.get("77", {})
    print(f"F{f}: T49 pos={t49_info.get('ground_pos')} spd={t49_info.get('speed_kmh')} | T74 pos={t74_info.get('ground_pos')} spd={t74_info.get('speed_kmh')} | T77 pos={t77_info.get('ground_pos')} spd={t77_info.get('speed_kmh')}")
