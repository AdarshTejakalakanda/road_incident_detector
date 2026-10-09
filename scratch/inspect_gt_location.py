import json

with open('scratch/z4_diagnostic_output.json') as f:
    data = json.load(f)

print("=== ALL CANDIDATES GENERATED ===")
for c in data['candidates']:
    sigs = c.get('signals', {})
    print(f"F{c['frame']} {c['event_type']} tracks={c['track_ids']} conf={c['confidence']} accel={sigs.get('accel_1_ms2')} iou={sigs.get('bbox_iou')} dist={sigs.get('ground_distance_m')} is_corridor={sigs.get('is_kinematic_closure_corridor')}")

print("\n=== DETECTIONS NEAR GT (x~900-1000, y~200-350) F120-F180 ===")
for f in range(120, 185):
    f_str = str(f)
    dets = data['detections_f150_f180'].get(f_str, [])
    if dets:
        for d in dets:
            bx = d['bbox']
            # center
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            if 800 <= cx <= 1250 and 150 <= cy <= 450:
                print(f"F{f} DET: conf={d['conf']} cls={d['cls']} center=({cx:.1f}, {cy:.1f}) bbox={bx}")
