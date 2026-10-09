import json, os

for f in os.listdir('artifacts'):
    if f.endswith('.json'):
        p = os.path.join('tifacts', f).replace('tifacts', 'artifacts')
        try:
            with open(p, 'r', encoding='utf-8') as js:
                data = json.load(js)
                if 'total_candidate_unconfirmed' in data:
                    print(f, 'unconfirmed:', data.get('total_candidate_unconfirmed'), 'total_eval:', data.get('total_evaluated'))
        except:
            pass
