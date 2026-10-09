import json

with open('artifacts/r_full_benchmark_sweep.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total evaluated: {data['total_evaluated']}, TPs: {data['metrics']['true_positives']}")
for r in data['per_video_results']:
    if r['is_tp']:
        vid = r['video_id']
        itype = r['incident_type']
        s_type = r['split_type']
        mc = r['matched_candidate']
        cand_f = mc['frame']
        cand_type = mc['event_type']
        sigs = mc['signals']
        
        acc = sigs.get('accel_1_ms2', sigs.get('metric_accel_ms2', sigs.get('decel_mag', 0.0)))
        dist = sigs.get('distance_m', sigs.get('d_edge_m', sigs.get('ground_distance_m', 0.0)))
        iou = sigs.get('bbox_iou', 0.0)
        heading_delta = sigs.get('heading_delta_deg', sigs.get('theta_deg', 0.0))
        rest = sigs.get('post_event_rest', False)
        v_res = r['vlm_tp_result']
        
        print(f"{vid:<16} | {itype:<8} | {s_type:<22} | F{cand_f:<3} | {cand_type:<22} | a={acc:>6.1f} | d={dist:>5.2f}m | iou={iou:>4.2f} | h_delta={heading_delta:>5.1f} | rest={rest} | vlm_src={v_res.get('source')}")
