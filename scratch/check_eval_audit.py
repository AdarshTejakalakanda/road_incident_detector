import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

print("eval_audit length:", len(eval_audit))
print("First item keys:", list(eval_audit[0].keys()))
print("First item:", json.dumps(eval_audit[0], indent=2))
