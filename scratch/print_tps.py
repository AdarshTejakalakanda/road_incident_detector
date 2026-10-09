import json, sys
sweep = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', 'r', encoding='utf-8'))
lines = []
for tp in sweep['per_video_results']:
    if tp['is_tp']:
        vid = tp['video_id']        mf = str(tp.get('matched_candidate', {}).get('frame', 'none'))
        btel = tp.get('vlm_tp_result', {}).get('bypass_telemetry', {})
        vel = btel.get('velocitiesW, {})
        imp = btel.get('impulse', {})
        cnt = btel.get('contact_evidence', {})
        rst = btel.get('rest_and_arrest', {})
        dv_drop = str(vel.get('speed_drop_kmh', 'none'))
        acc = str(imp.get('accel_ms2', 'none'))
        dist = str(cnt.get('distance_m', 'none'))
        iou = str(cnt.get('bbox_iou', 'none'))
        npx = str(cnt.get('norm_pixel_dist', 'none'))
        tau_r = str(rst.get('tau_rest', 'none'))
        breason = str(btel.get('bypass_reason', 'none'))
        lines.append(vid.ljust(15) + ' | F' + mf.ljust(5) + ' | dv:' + dv_drop.ljust(5) + ' | a:' + acc.ljust(6) + ' | dist:' + dist.ljust(6) + ' | iou:' + iou.ljust(5) + ' | npx:' + npx.ljust(5) + ' | tau:' + tau_r.ljust(5) + ' | ' + breason)

open('scratch/tp_details.txt', 'w', encoding='utf-8').write('\n'.join(lines))
print('WROTE TP')
