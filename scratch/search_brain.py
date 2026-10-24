import os
import sys

base_dir = r"C:\Users\Adarsh Teja Kalakand\.gemini\antigravity\brain\7022a033-0dbc-4c26-bbd6-d3313704db4f"
terms = ["Policy 1", "Policy 2", "E_P1", "E_P2", "Policy1", "Policy2"]

for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith(('.json', '.md', '.txt', '.py', '.log', '.csv')):
            p = os.path.join(root, f)
            if 'transcript' in f:
                continue
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    for lno, line in enumerate(fp, 1):
                        if any(t in line for t in terms):
                            print(f"{p}:{lno}: {line.strip()[:120]}")
            except Exception:
                pass
