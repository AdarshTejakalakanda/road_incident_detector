import json

with open('scratch/parsed_71_fps.json', 'r', encoding='utf-8') as f:
    all_fps = json.load(f)

noc_vids = {'ftk_fSc4haM_00', 'Z4kg2Ev3vhk_00', 'UarP8qU1S-c_00'}
noc_fps = [x for x in all_fps if x['video_id'] in noc_vids]

out = []
out.append('Total Nocturnal FPs in parsed data: ' + str(len(noc_fps)))

for i, x in enumerate(noc_fps):
    vid = x['video_id']
    f = str(x['frame'])
    r = x['reason_base']
    cnt = x['contact']
    imp = x['impulse']
    iou = str(round(cnt.get('bbox_iou', 0.0), 3))
    dist = str(round(cnt.get('distance_m', 0.0), 3))
    norm_px = str(round(cnt.get('norm_pixel_dist', 0.0), 3))
    accel = str(round(imp.get('accel_ms2', 0.0), 1))
    s_dissip = str(round(cnt.get('S_dissip', 0.0), 2))
    out.append(str(i+1).rjust(2) + '. ' + vid.ljust(15) + ' @ F' + f.rjust(3) + ' | Rule: ' + r)
    out.append('    Telemetry: IoU=' + iou + ', dist_m=' + dist + ', norm_px=' + norm_px + ', accel=' + accel + ', S_dissip=' + s_dissip)

with open('scratch/noc_audit_out.txt', 'w', encoding='utf-8') as f_out:
    f_out.write('\n'.join(out))