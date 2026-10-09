with open('tools/benchmark_sa_full_sweep.py') as f:
    lines = f.readlines()


for i, l in enumerate(lines):
    if '[FP CONFIRMED]' in l:
        print(f'Line {i+1}: ', l)
        print(''.join(lines[max(0, i-5):min(len(lines), i+15)]))
