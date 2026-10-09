with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.splitlines()
for i in range(485, 505):
    print(f"{i+1}: {lines[i]}")
