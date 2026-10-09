with open('scratch/fp_lines.txt', 'r', encoding='utf-8', errors='ignore') as f:
    lines = [l.strip() for l in f if any(v in l for v in ['Z4kg2Ev3vhk_00', 'ftk_fSc4haM_00', 'UarP8qU1S-c_00'])]

with open('scratch/noc_lines.txt', 'w', encoding='utf-8') as out:
    out.write('\n'.join(lines))