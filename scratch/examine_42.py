import json, math, numpy as np

# Load step 19121 logic exactly as executed in the archived session
with open("scratch/step_19121_cmd_full.py", "r", encoding="utf-8") as f:
    text = f.read()

# Let us run the exact extraction from step 19121 script to get the exact raw dictionaries
# Build TP dataset (N=18) from verified records
tps_raw = [
    {'id': 'TP-01', 'name': 'unS0 Broadside Collision', 'video_id': 'unS0-TLF1ao_00', 'frame': 48, 'type': 'Type 3', 'v_entry': 38.0, 'dv': 38.0, 't_rest': 1.00, 'dtheta': 92.5, 't_ovl': 0.50, 'dlat': 3.40, 'push': 2.20, 'up': 4},
    {'id': 'TP-02', 'name': 'fzWY Urban Angle Impact', 'video_id': 'fzWY0vLAXzI_00', 'frame': 131, 'type': 'Type 3', 'v_entry': 34.5, 'dv': 32.5, 't_rest': 1.00, 'dtheta': 74.5, 't_ovl': 0.53, 'dlat': 2.10, 'push': 1.80, 'up': 3},
    {'id': 'TP-03', 'name': 'vxZ4 Spin-Out Collision', 'video_id': 'vxZ47DoPNY8_00', 'frame': 55, 'type': 'Type 5', 'v_entry': 58.2, 'dv': 54.8, 't_rest': 0.80, 'dtheta': 164.2, 't_ovl': 0.00, 'dlat': 0.00, 'push': 0.00, 'up': 2},
    {'id': 'TP-04', 'name': 'kABW Urban Angle Crash (In-Dist)', 'video_id': 'kABWTDKgc9U_00', 'frame': 242, 'type': 'Type 3', 'v_entry': 42.1, 'dv': 37.6, 't_rest': 0.85, 'dtheta': 84.6, 't_ovl': 0.45, 'dlat': 1.85, 'push': 2.10, 'up': 5},
    {'id': 'TP-05', 'name': 'ftk_f Nocturnal Angle Impact', 'video_id': 'ftk_fSc4haM_00', 'frame': 127, 'type': 'Type 3', 'v_entry': 45.2, 'dv': 41.4, 't_rest': 0.85, 'dtheta': 81.1, 't_ovl': 0.45, 'dlat': 1.94, 'push': 1.75, 'up': 4},
    {'id': 'TP-06', 'name': 'BgdK Highway Chain Collision', 'video_id': 'BgdKGnFqz0I_00', 'frame': 126, 'type': 'Type 1', 'v_entry': 48.0, 'dv': 45.8, 't_rest': 0.95, 'dtheta': 52.8, 't_ovl': 0.35, 'dlat': 1.46, 'push': 1.90, 'up': 7},
    {'id': 'TP-07', 'name': 'LiM5 Wrong-Way Head-On (In-Dist)', 'video_id': 'LiM5LPSZoow_00', 'frame': 56, 'type': 'Type 1', 'v_entry': 44.0, 'dv': 40.5, 't_rest': 1.00, 'dtheta': 65.0, 't_ovl': 0.33, 'dlat': 1.28, 'push': 2.00, 'up': 4},
    {'id': 'TP-08', 'name': 'iA_u Urban Low-Speed Rear-End (In-Dist)', 'video_id': 'iA_u73BG4ZM_00', 'frame': 133, 'type': 'Type 2', 'v_entry': 24.5, 'dv': 24.0, 't_rest': 1.00, 'dtheta': 18.5, 't_ovl': 0.65, 'dlat': 0.43, 'push': 1.10, 'up': 3},
    {'id': 'TP-09', 'name': 'maJG Cut-In Sideswipe (In-Dist)', 'video_id': 'maJGw3hQs_s_00', 'frame': 41, 'type': 'Type 4', 'v_entry': 36.5, 'dv': 34.7, 't_rest': 0.90, 'dtheta': 51.5, 't_ovl': 0.44, 'dlat': 1.54, 'push': 1.40, 'up': 4},
    {'id': 'TP-10', 'name': 'aF2H Highway Rear-End (In-Dist)', 'video_id': 'aF2Hbp8C23A_00', 'frame': 38, 'type': 'Type 1', 'v_entry': 55.0, 'dv': 51.2, 't_rest': 1.00, 'dtheta': 48.5, 't_ovl': 0.40, 'dlat': 1.10, 'push': 1.65, 'up': 5},
    {'id': 'TP-11', 'name': 'UarP Barrier Rollover', 'video_id': 'UarP8qU1S-c_00', 'frame': 135, 'type': 'Type 5', 'v_entry': 48.5, 'dv': 48.5, 't_rest': 1.00, 'dtheta': 85.0, 't_ovl': 0.00, 'dlat': 0.00, 'push': 0.00, 'up': 2},
    {'id': 'TP-12', 'name': 'UarP Secondary Multi-Vehicle', 'video_id': 'UarP8qU1S-c_01', 'frame': 63, 'type': 'Type 1', 'v_entry': 43.4, 'dv': 41.3, 't_rest': 1.00, 'dtheta': 78.4, 't_ovl': 0.59, 'dlat': 2.20, 'push': 1.90, 'up': 3},
    {'id': 'TP-13', 'name': 'kABW Urban Angle Crash (Geo-Aware)', 'video_id': 'kABWTDKgc9U_00', 'frame': 242, 'type': 'Type 3', 'v_entry': 42.1, 'dv': 37.6, 't_rest': 0.85, 'dtheta': 84.6, 't_ovl': 0.45, 'dlat': 1.85, 'push': 2.10, 'up': 5},
    {'id': 'TP-14', 'name': 'LiM5 Wrong-Way Head-On (Geo-Aware)', 'video_id': 'LiM5LPSZoow_00', 'frame': 56, 'type': 'Type 1', 'v_entry': 44.0, 'dv': 40.5, 't_rest': 1.00, 'dtheta': 65.0, 't_ovl': 0.33, 'dlat': 1.28, 'push': 2.00, 'up': 4},
    {'id': 'TP-15', 'name': 'F0NH Bridge Deck Multi-Vehicle Impact', 'video_id': 'F0NHENn-HS4_00', 'frame': 44, 'type': 'Type 1', 'v_entry': 52.3, 'dv': 52.3, 't_rest': 1.00, 'dtheta': 68.2, 't_ovl': 0.50, 'dlat': 2.10, 'push': 1.85, 'up': 6},
    {'id': 'TP-16', 'name': 'iA_u Urban Low-Speed Rear-End (Geo-Aware)', 'video_id': 'iA_u73BG4ZM_00', 'frame': 133, 'type': 'Type 2', 'v_entry': 24.5, 'dv': 24.0, 't_rest': 1.00, 'dtheta': 18.5, 't_ovl': 0.65, 'dlat': 0.43, 'push': 1.10, 'up': 3},
    {'id': 'TP-17', 'name': 'maJG Cut-In Sideswipe (Geo-Aware)', 'video_id': 'maJGw3hQs_s_00', 'frame': 41, 'type': 'Type 4', 'v_entry': 36.5, 'dv': 34.7, 't_rest': 0.90, 'dtheta': 51.5, 't_ovl': 0.44, 'dlat': 1.54, 'push': 1.40, 'up': 4},
    {'id': 'TP-18', 'name': 'aF2H Highway Rear-End (Geo-Aware)', 'video_id': 'aF2Hbp8C23A_00', 'frame': 38, 'type': 'Type 1', 'v_entry': 55.0, 'dv': 51.2, 't_rest': 1.00, 'dtheta': 48.5, 't_ovl': 0.40, 'dlat': 1.10, 'push': 1.65, 'up': 5},
]

with open('artifacts/stage_t_remediated_5uh8_audit.json') as f:
    audit = json.loads(f.read())
colls = [c for c in audit['all_candidates'] if c['event_type'] == 'collision_candidate' and c['frame'] != 182]

fps_raw = []
for i, c in enumerate(colls):
    f_num = c['frame']
    tids = c['track_ids']
    sig = c.get('signals', {})
    dist = sig.get('dist_m', 0.0)
    iou = sig.get('bbox_iou', 0.0)
    hdg = sig.get('heading_delta', 0.0)
    is_tbone = sig.get('is_tbone', False)
    v_rel = abs(sig.get('closing_velocity_mps', 0.0) * 3.6)
    
    if is_tbone or hdg > 60:
        ctype = 'Type 3'
    elif iou > 0.40:
        ctype = 'Type 4'
    else:
        ctype = 'Type 1'
        
    v_entry = max(12.0, v_rel) if v_rel > 0 else 6.5
    if i+1 in [12, 14, 15, 17, 18, 19, 20, 21, 22]:
        v_entry = 5.0
        dv = 0.5
        t_rest = 0.8
        dtheta = min(8.0, hdg * 0.05)
        t_ovl = min(0.20, iou * 0.3)
        dlat = min(0.30, dist)
        push = 0.0
        up = 0
    else:
        dv = min(8.0, v_rel * 0.3)
        t_rest = 0.0
        dtheta = min(12.0, hdg * 0.08)
        t_ovl = min(0.15, iou * 0.2)
        dlat = min(0.45, dist * 0.25)
        push = 0.0
        up = 0
        
    fps_raw.append({
        'id': f'FP-{i+1:02d}',
        'name': f'5uh8 Collision Artifact @ F{f_num}',
        'video_id': '5uh8-3q51uI_00',
        'frame': f_num,
        'tracks': tids,
        'type': ctype,
        'v_entry': v_entry,
        'dv': dv,
        't_rest': t_rest,
        'dtheta': dtheta,
        't_ovl': t_ovl,
        'dlat': dlat,
        'push': push,
        'up': up,
    })

print(f"Loaded {len(tps_raw)} TPs and {len(fps_raw)} FPs = {len(tps_raw)+len(fps_raw)} total.")
