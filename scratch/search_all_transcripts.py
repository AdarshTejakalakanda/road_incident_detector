import os
import re
import json

brain_dir = r"C:\Users\Adarsh Teja Kalakand\.gemini\antigravity\brain"

for cid in os.listdir(brain_dir):
    cpath = os.path.join(brain_dir, cid)
    if not os.path.isdir(cpath):
        continue
    log_file = os.path.join(cpath, ".system_generated", "logs", "transcript.jsonl")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Policy 1" in line or "Policy 2" in line or "E_P1" in line:
                        obj = json.loads(line)
                        src = obj.get("source")
                        content = obj.get("content", "")
                        for l in content.split("\n"):
                            if any(k in l for k in ["Policy 1", "Policy 2", "E_P1", "E_P2"]):
                                print(f"[{cid[:8]}] {src}: {l[:140]}")
        except Exception:
            pass
