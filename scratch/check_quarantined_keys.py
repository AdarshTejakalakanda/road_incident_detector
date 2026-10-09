import json

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp = json.load(f)

print("First item keys:", list(supp[0].keys()))
print("First item content:", json.dumps(supp[0], indent=2))
