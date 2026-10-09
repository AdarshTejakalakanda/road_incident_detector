with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_func = False
func_lines = []
for i, l in enumerate(lines):
    if '_evaluate_deterministic_high_confidence_bypass' in l:
        in_func = True
    if in_func:
        func_lines.append((i + 1, l))
        if 'def ' in l and '_evaluate_deterministic_high_confidence_bypass' not in l:
            break

with open('scratch/bypass_func_lines.txt', 'w', encoding='utf-8') as out:
    for num, l in func_lines:
        out.write(f'{num:4d}: {l}')

print('Extracted bypass function lines:', len(func_lines))