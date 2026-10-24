import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
path = r'C:/Users/Adarsh Teja Kalakand/.gemini/antigravity/brain/7022a033-0dbc-4c26-bbd6-d3313704db4f/.system_generated/logs/transcript_full.jsonl'
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        idx = obj.get('step_index')
        if idx in [2045, 2046, 2048, 2050, 2056, 2076, 2087, 2138]:
            print(f"=== Step {idx} (type={obj.get('type')}, source={obj.get('source')}) ===")
            if obj.get('content'):
                print("CONTENT:", obj.get('content')[:500])
            if obj.get('thinking'):
                print("THINKING:", obj.get('thinking')[:500])
            if obj.get('tool_calls'):
                print("TOOL_CALLS:", obj.get('tool_calls'))
