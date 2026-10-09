import re, json
from collections import Counter

with open('scratch/fp_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [l.strip() for l in f if l.strip()]

records = []
reason_counts = Counter()
vid_counts = Counter()

# Pattern example:
# [FP CONFIRMED] Z4kg2Ev3vhk_00 @ F113 (collision_candidate): reason=multi_vehicle_contact_impulse (dist=0.96, iou=0.25, accel=-60.0) | contact={'bbox_iou': 0.25, 'distance_m': 0.963, 'norm_pixel_dist': 0.822, 'is_crossing': False, 'is_kinematic_closure_corridor': False, 'S_dissip': 0.7} | impulse={'accel_ms2': -60.0, 'abs_accel_ms2': 60.0}

pattern = re.compile(r'\[FP CONFIRMED\]\s+([^\s]+)\s+@\s+F(\d+)\s+\(([^)]+)\):\s+reason=([^\s|]+(?:\s+\([^)]+\))?)\s+\|\s+contact=([^|]+)\s+\|\s+impulse=(.+)')

for l in lines:
    m = pattern.search(l)
    if m:
        vid, frame, ctype, reason_full, contact_str, impulse_str = m.groups()
        reason_base = reason_full.split('(')[0].strip()
        reason_counts[reason_base] += 1
        vid_counts[vid] += 1
        
        # parse contact
        try:
            contact_dict = eval(contact_str.strip())
        except:
            contact_dict = {}
        try:
            impulse_dict = eval(impulse_str.strip())
        except:
            impulse_dict = {}
            
        records.append({
            'video_id': vid,
            'frame': int(frame),
            'event_type': ctype,
            'reason_full': reason_full,
            'reason_base': reason_base,
            'contact': contact_dict,
            'impulse': impulse_dict
        })
    else:
        print('UNPARSED LINE:', l)

print(f'Parsed {len(records)} / {len(lines)} records')
print('\nREASON DISTRIBUTION:')
for r, c in reason_counts.most_common():
    print(f'  {r:35s}: {c:2d} ({c/len(records)*100:.1f}%)')

print('\nPER-VIDEO FP COUNTS:')
for v, c in vid_counts.most_common():
    print(f'  {v:20s}: {c:2d}')

with open('scratch/parsed_71_fps.json', 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=2)
