import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:/Users/Adarsh Teja Kalakand/.gemini/antigravity/brain/7022a033-0dbc-4c26-bbd6-d3313704db4f/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'Policy 1' in line or 'Policy 2' in line or 'E_P1' in line:
            obj = json.loads(line)
            c = obj.get('content', '')
            if 'Policy 1' in c or 'E_P1' in c or 'E_{P1}' in c:
                print(f"Step {obj.get('step_index')} by {obj.get('source')}:")
                for sub in c.split('\n'):
                    if any(term in sub for term in ['Policy 1', 'Policy 2', 'E_P1', 'E_P2', 'E_{P1}', 'E_{P2}', 'Policy']):
                        print('  ', sub.strip()[:140])
