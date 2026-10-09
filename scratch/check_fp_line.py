with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'has_finite_dissipation' in line or 'has_unimputed_rest' in line:
        print(f"{i+1}: {line.strip()}")
