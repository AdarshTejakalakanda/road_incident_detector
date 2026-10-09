with open('artifacts/stage2_nocturnal_regression.json', 'r', encoding='utf-8') as f:
    import json
    d = json.load(f)

for r in d.get('results', []):
    if r['video_id'] == 'ftk_fSc4haM_00':
        for ep in r['episodes']:
            if 'f27' in ep or 'f26' in ep or 'f34' in ep:
                print('HISTORICAL EP:', ep[:300])