import json

with open('scratch/z4_diagnostic_output.json') as f:
    data = json.load(f)

for f in range(168, 173):
    print(f'=== FRAME {f} ===')
    dets = data['detections_f150_f180'].get(str(f), [])
    print(f'  Detections ({len(dets)}):')
    for d in dets:
        print(f'    cls={d["cls"]} conf={d["conf"]} bbox={d["bbox"]}')
    tracks = data['tracks_f150_f180'].get(str(f), [])
    print(f'  Tracks ({len(tracks)}):')
    for t in tracks:
        if t['id'] in [7, 10, 11, 14, 17, 19, 20, 21, 22]:
            print(f'    ID {t["id"]} ({t["label"]}) age={t["age"]} spd={t["speed_kmh"]} hdg={t["heading_deg"]} acc={t["accel_ms2"]} box={t["bbox"]}')
