import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:/Users/Adarsh Teja Kalakand/.gemini/antigravity/brain/7022a033-0dbc-4c26-bbd6-d3313704db4f/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get('step_index')
        if 45 <= idx <= 58:
            print(f"=== Step {idx} (src={obj.get('source')}, type={obj.get('type')}) ===")
            c = obj.get('content', '')
            if c:
                print(c[:1000])
