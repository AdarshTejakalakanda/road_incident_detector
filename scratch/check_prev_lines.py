with open('modules/vlm_arbitrator.py.bak', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(468, 476):
    print(str(i+1) + ': ' + lines[i].strip())
