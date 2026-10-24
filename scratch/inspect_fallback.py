import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

fallback = [ep for ep in eval_audit if ep.get("disposition") == "FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW"]
print(f"Fallback count: {len(fallback)}")
for ep in fallback:
    print(json.dumps(ep, indent=2))
