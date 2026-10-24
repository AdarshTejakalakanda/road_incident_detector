import os
import sys

terms = ['Policy 1', 'Policy 2', 'E_P1', 'E_P2', 'EP1', 'EP2', 'policy 1', 'policy 2']

for root, dirs, files in os.walk('.'):
    if any(p in root for p in ['.git', '.venv', '__pycache__']):
        continue
    for file in files:
        if file.endswith(('.json', '.py', '.txt', '.md', '.csv')):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_no, line in enumerate(f, 1):
                        if any(t in line for t in terms):
                            print(f"{filepath}:{line_no}: {line.strip()[:100]}")
            except Exception:
                pass
