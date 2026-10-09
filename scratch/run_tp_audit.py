import json
f = open('artifacts/post_ab_full_benchmark_sweep.json', 'r', encoding='utf-8')
sweep = json.load(f)
pvr = sweep['per_video_results']
tp_records = [x for x in pvr if x['is_tp']]
lines = ['TP count: ' + str(len(tp_records))]
for tp in tp_records:
    vid = tp['video_id']
    match = tp.get('matched_candidate')
    vlm_tp = tp.get('vlm_tp_result') or {}
    btel = vlm_tp.get('bypass_telemetry') or {}
    breason = btel.get('bypass_reason', 'none')
    src = vlm_tp.get('source', 'none')
    mf = str(match.get('frame')) if match else 'none'
    lines.append(vid.ljust(16) + ' | F' + mf.ljust(5) + ' | src: ' + src.ljust(35) + ' | reason: ' + breason)
f_out = open('scratch/tp_bypass_audit.txt', 'w', encoding='utf-8')
f_out.write('\n'.join(lines))
f_out.close()
print('WROTE TM OK')
