import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['is_tp']:
        vid = p['video_id']
        mat = p.get('matched_candidate', {})
        sigs = mat.get('signals', {})
        print(f"{vid} | dist={sigs.get('distance_m')} | norm_px={sigs.get('norm_pixel_dist')} | iou={sigs.get('bbox_iou')} | s_diss={sigs.get('S_dissip')} | post_rest={sigs.get('post_event_rest')}")
