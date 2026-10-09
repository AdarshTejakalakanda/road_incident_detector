with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
found1 = False
found2 = False
for i, line in enumerate(lines):
    if '"track_age": 10,' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        calc_line = indent + '"track_age": min([len(track_life_history.get(tid, [])) for tid in cand.get("track_ids", []*]) if cand.get("track_ids") else 0,\n'
        new_lines.append(calc_line)
        found1 = True
        print('Replaced line', i+1, '(in-window track_age)')
    elif '"track_age": 4,' in line:
        indent = ' ' * (len(line) - len(line.lstrip()))
        calc_line = indent + '"track_age": min([len(track_life_history.get(tid, [])) for tid in fp_cand.get("track_ids", [])]) if fp_cand.get("track_ids") else 0,\n'
        new_lines.append(calc_line)
        found2 = True
        print('Replaced line', i+1, '(out-of-window track_age)')
    else:
        new_lines.append(line)

assert found1, 'target1 not found'
assert found2, 'target2 not found'

with open('tools/benchmark_sa_full_sweep.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Updated tools/benchmark_sa_full_sweep.py successfully!')
