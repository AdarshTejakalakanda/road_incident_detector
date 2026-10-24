import json

path_eval = "artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json"
path_supp = "artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json"

with open(path_eval, "r", encoding="utf-8") as f:
    eval_obj = json.load(f)

with open(path_supp, "r", encoding="utf-8") as f:
    supp_obj = json.load(f)

print(f"eval_obj type: {type(eval_obj)}")
print(f"supp_obj type: {type(supp_obj)}")
