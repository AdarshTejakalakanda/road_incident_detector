import json

with open('scratch/z4_diagnostic_output.json') as f:
    data = json.load(f)

print('=== CANDIDATES IN WHOLE VIDEO ===')
for c in data['candidates']:
    sigs = c.get('signals', {})
    print(f"F{c['frame']} {c['event_type']} tracks={c['track_ids']} conf={c['confidence']} spd={sigs.get('metric_speed_kmh')} acc={sigs.get('accel_1_ms2')} iou={sigs.get('bbox_iou')} dist={sigs.get('ground_distance_m')}")

print('\n=== EPISODES ===')
for ep in data['episodes']:
    print(f"Episode {ep['ep_id']}: peak=F{ep['peak_f']} range={ep['range']} tracks={ep['tracks']} primary={ep['primary_pair']} in_window={ep['in_window']} confirmed={ep['confirmed']} verdict={ep['verdict']}")
    print(f"   Rationale: {ep['rationale']}")

print('\n=== TRACKS AT F150, F155, F160, F165, F170, F175, F180 ===')
for f in [150, 155, 160, 165, 170, 175, 180]:
    tracks = data['tracks_f150_f180'].get(str(f), [])
    print(f'Frame {f}: {len(tracks)} tracks -> {[t["id"] for t in tracks]}')
    for t in tracks:
        print(f'   ID {t["id"]} ({t["label"]}) age={t["age"]} spd={t["speed_kmh"]} hdg={t["heading_deg"]} acc={t["accel_ms2"]} box={t["bbox"]}')

print('\n=== RAW YOLO ADMITTED DETECTIONS IN F165-F175 ===')
for f in range(165, 176):
    dets = data['detections_f150_f180'].get(str(f), [])
    print(f'Frame {f} ({len(dets)} dets):')
    for d in dets:
        print(f'   conf={d["conf"]} cls={d["cls"]} bbox={d["bbox"]}')

print('\n=== PAIR EVALS IN F150-F180 ===')
for f in range(150, 181):
    evals = data['pair_evals_f150_f180'].get(str(f), [])
    if evals:
        print(f'Frame {f} ({len(evals)} evals):')
        for e in evals:
            print(f'   Pair {e["pair"]}: dist={e["dist_m"]}m dh={e["dh_deg"]} iou={e["iou"]} norm_px={e["norm_pixel_dist"]} accels={e["accels"]} c_cnt={e["closure_cnt"]} contact={e["has_contact"]} decel={e["has_decel"]} sanity={e["has_sanity"]}')

print('\n=== SUPPRESSED IN F150-F180 ===')
for f, supps in data['suppressed_f150_f180'].items():
    print(f'Frame {f}: {supps}')
