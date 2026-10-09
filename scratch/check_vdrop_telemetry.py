with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Let us see how max_recent_speed_kmh and delta_v_drop are computed in the sweep harness
for i, l in enumerate(text.splitlines()):
    if 'max_recent_speed_kmh' in l or 'speed_drop' in l or 'delta_v' in l:
        print(f'{i+1:4d}: {l}')