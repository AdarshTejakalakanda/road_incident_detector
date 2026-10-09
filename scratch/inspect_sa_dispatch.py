with open('tools/benchmark_sa_full_sweep.py') as f:
    lines = f.readlines()


for i, l in enumerate(lines):
    if 'cand_res = arbitrator.arbitrate' in l or 'fp_res = arbitrator.arbitrate' in l:
        print(f'Line {i+1}:')
        print(''.join(lines[max(0, i-5):min(len(lines), i+35)]))
