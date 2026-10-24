import os
import re

for fname in os.listdir('scratch'):
    if fname.endswith(('.py', '.md', '.txt', '.json')):
        p = os.path.join('scratch', fname)
        if os.path.getsize(p) > 500000:
            continue
        try:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                m = re.findall(r'(\b(?:Policy\s*[12]|E_P[12]|P[12]_score|policy[12])\b)', content, re.IGNORECASE)
                if m:
                    print(f"Found {m} in {fname}")
        except Exception:
            pass
