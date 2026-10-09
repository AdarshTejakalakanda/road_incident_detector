with open("scratch/step_19121_cmd_full.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")
for i in range(25, 46):
    if i < len(lines):
        print(f"{i}: {lines[i]}")
