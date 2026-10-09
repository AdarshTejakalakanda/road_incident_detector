with open('artifacts/stage2_nocturnal_regression.json', 'r', encoding='utf-8') as f:
    import json
    d = json.load(f)

for r in d.get('results', []):
    if r['video_id'] == 'ftk_fSc4haM_00':
        for ep in r['episodes']:
            if 'ep08_f34' in ep:
                print('HISTORICAL EP08_F34:')
                idx_res = ep.find('vlm_result=')
                print(ep[idx_res:idx_res+300])