with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Check all places where tau_rest is computed or used
for i, l in enumerate(text.splitlines()):
    if 'tau_rest' in l:
        print(f'{i+1:4d}: {l}')