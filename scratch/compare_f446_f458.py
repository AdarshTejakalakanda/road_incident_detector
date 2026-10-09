"""Compare F446 [113, 127] and F458 [113, 127].
Extract bounding boxes, coordinates, speeds, and provenance for both tracks across F440-F465.
"""

import json

with open("scratch/forensic_10_records_data.json") as f:
    data = json.load(f)

r6 = data["records"]["6"]
t113 = r6["t1_hist"]
t127 = r6["t2_hist"]

print("--- Detailed Comparison of Tracks 113 and 127 around F446 and F458 ---")
for f in range(440, 465):
    f_str = str(f)
    s113 = t113.get(f_str)
    s127 = t127.get(f_str)
    if s113 and s127:
        b1 = s113['bbox']
        b2 = s127['bbox']
        # Compute 2D IoU
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[0] + b1[2], b2[0] + b2[2])
        yB = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        iou = inter / max(1e-6, area1 + area2 - inter)
        d_lat = abs(s113['ground_pos'][0] - s127['ground_pos'][0])
        d_long = abs(s113['ground_pos'][1] - s127['ground_pos'][1])
        print(f"F{f}: IoU={iou:.2f}, d_lat={d_lat:.2f}m, d_long={d_long:.2f}m | T113 bbox={b1} spd={s113['speed_kmh']} | T127 bbox={b2} spd={s127['speed_kmh']}")
