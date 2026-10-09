import json, os

for f in os.listdir('.'):
    if f.startswith('telemetry_') and f.endswith('.json'):
        with open(f, 'r', encoding='utf-8') as js:
            data = json.load(js)
            print(f, 'type:', type(data), 'len:', len(data))
            if isinstance(data, dict):
                print('  keys:', list(data.keys())[:5])
