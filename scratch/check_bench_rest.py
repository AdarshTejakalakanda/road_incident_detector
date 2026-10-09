with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(490, 535):
    print(str(i+1) + ': ' + lines[i].strip())
