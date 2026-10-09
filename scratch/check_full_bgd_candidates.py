import json
with open('artifacts/post_ab_full_benchmark_sweep.json', 'r') as f:
    data = json.load(f)

for p in data['per_video_results']:
    if p['video_id'] == 'BgdKGnFqz0I_00':
        mat = p.get('matched_candidate', {})
        print('mat frame:', mat.get('frame'))
        sigs = mat.get('signals', {})
        print('mat accel_1_ms2:', sigs.get('accel_1_ms2'))
        print('mat accel_2_ms2:', sigs.get('accel_2_ms2'))
        print('mat distance_m:', sigs.get('distance_m'))
        print('mat norm_pixel_dist:', sigs.get('norm_pixel_dist'))
        print('mat bbox_iou:', sigs.get('bbox_iou'))
        print('mat S_dissip:', sigs.get('S_dissip'))
        print('mat post_event_rest:', sigs.get('post_event_rest'))
