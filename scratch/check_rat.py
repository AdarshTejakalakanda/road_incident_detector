import json
d = json.load(open('artifacts/stage2_nocturnal_regression.json', encoding='utf-8'))
for r in d['results']:
    if r['video_id'] == 'ftk_fSc4haM_00':
        for ep in r['episodes']:
            idx = ep.find('rationale')
            if idx != -1:
                print(ep[idx:idx+120])