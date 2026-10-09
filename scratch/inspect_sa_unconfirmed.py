with open('tools/benchmark_sa_full_sweep.py') as f:
    lines = f.readlines()


for i, l in enumerate(lines):
    if 'candidate_unconfirmed_count +=' in l:
        print(f'Line {i+1}: {l.strip()}')
        print(''.join(lines[max(0, i-8):min(len(lines), i+8)]))
