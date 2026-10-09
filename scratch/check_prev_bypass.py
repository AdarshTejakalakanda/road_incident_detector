with open('modules/vlm_arbitrator.py.bak', 'r', encoding='utf-8') as f:
    text = f.read()

for line in text.splitlines():
    if 'stage_t_deterministic_bypass' in line or 'tau_rest = 0.25' in line:
        print(line.strip())
