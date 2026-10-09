import os

for f in os.listdir('scratch'):
    if f.endswith('.py'):
        p = os.path.join('scratch', f)
        with open(p, 'r', encoding='utf-8', errors='ignore') as file:
            t = file.read()
            if 'benchmark_sa_full_sweep' in t or 'sa_full_benchmark' in t:
                print('Found in scratch:', f)
