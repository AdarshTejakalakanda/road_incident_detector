import json

with open('artifacts/stage2_nocturnal_regression.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for r in d.get('results', []):
    vid = r['video_id']
    if vid in ['ftk_fSc4haM_00', 'Z4kg2Ev3vhk_00', 'UarP8qU1S-c_00']:
        print(vid + ' | TP: ' + str(r['is_tp']) + ' | FP: ' + str(r['fp_confirmed_count']) + ' | OutOfWinEps: ' + str(r['out_of_window_episodes_count']))