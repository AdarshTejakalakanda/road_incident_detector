with open('tools/benchmark_sa_full_sweep.py') as f:
    lines = f.readlines()


for i, l in enumerate(lines):
    if l.startswith('def '):
        print(f"Line {i+1}: {l.strip()}")
