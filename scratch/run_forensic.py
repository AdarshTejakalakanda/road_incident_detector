import json

with open('scratch/parsed_71_fps.json', 'r', encoding='utf-8') as f:
    fps = json.load(f)

out = []
out.append('Total parsed FPs: ' + str(len(fps)))
by_rule = {}
for fp in fps:
    r = fp['reason_base']
    by_rule.setdefault(r, []).append(fp)

for r, items in by_rule.items():
    ious = [it['contact'].get('bbox_iou', 0.0) for it in items]
    accels = [it['impulse'].get('accel_ms2', 0.0) for it in items]
    dists = [it['contact'].get('distance_m', 99.0) for it in items]
    vids = sorted(list(set(it['video_id'] for it in items)))
    out.append('RULE: ' + r + ' | N: ' + str(len(items)))
    out.append('  Videos: ' + str(vids))
    out.append('  IoU: [' + str(round(min(ious), 2)) + ', ' + str(round(sum(ious)/len(ious), 2)) + ', ' + str(round(max(ious), 2)) + ']')
    out.append('  accel: [' + str(round(min(accels), 1)) + ', ' + str(round(sum(accels)/len(accels), 1)) + ', ' + str(round(max(accels), 1)) + ']')
    out.append('  dist_m: [' + str(round(min(dists), 2)) + ', ' + str(round(sum(dists)/len(dists), 2)) + ', ' + str(round(max(dists), 2)) + ']')

with open('scratch/forensic_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))