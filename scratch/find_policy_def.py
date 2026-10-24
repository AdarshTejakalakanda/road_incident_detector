import json
import re

path = r'C:/Users/Adarsh Teja Kalakand/.gemini/antigravity/brain/7022a033-0dbc-4c26-bbd6-d3313704db4f/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        c = obj.get('content', '') + ' ' + obj.get('thinking', '')
        # Search for Policy 1 / Policy 2 definitions
        for match in re.finditer(r'(?:Policy\s*[12]|E_\{?P[12]\}?|policy\s*[12])', c):
            start = max(0, match.start() - 200)
            end = min(len(c), match.end() + 200)
            snippet = c[start:end].replace('\n', ' ')
            step = obj.get('step_index')
            src = obj.get('source')
            print(f"Step {step} ({src}): ... {snippet} ...\n")
