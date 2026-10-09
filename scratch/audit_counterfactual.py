import json

with open(''artifacts/post_ab_full_benchmark_sweep.json'', ''r'', encoding=''utf-8'') as f:
    sweep = json.load(f)

pvr = sweep[''per_video_results'']
print(''Total videos in sweep: '' + str(len(pvr)))

tp_records = [x for x in pvr if x[''is_tp'']]
print(''TP records count: '' + str(len(tp_records)))
for tp in tp_records:
    vid = tp[''video_id'']
    match = tp.get(''matched_candidate'')
    vlm_tp = tp.get(''vlm_tp_result'')
    source = vlm_tp.get(''source'') if vlm_tp else ''None''
    reason = vlm_tp.get(''rationale'') if vlm_tp else ''None''
    btel = vlm_tp.get(''bypass_telemetry'', {}) if vlm_tp else {}
    breason = btel.get(''bypass_reason'', ''None'')
    print(vid + '' | matched: '' + str(match.get(''frame'') if match else None) + '' | source: '' + str(source) + '' | reason: '' + str(breason))