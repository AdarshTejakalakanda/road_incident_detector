import json

# Let us run the exact extraction from step 19121 to get all raw and computed values
with open("scratch/examine_42.py") as f:
    code = f.read()

# Let us re-import the definitions from examine_42.py
exec(code)

print("--- Sample TP-01 ---")
print(tps_raw[0])
print("--- Sample FP-01 ---")
print(fps_raw[0])
print("--- Sample FP-11 ---")
print(fps_raw[10])
print("--- Sample FP-24 ---")
print(fps_raw[23])
