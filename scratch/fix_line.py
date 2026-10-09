with open('tools/benchmark_sa_full_sweep.py', 'rb') as f:
    content = f.read()

content = content.replace(b'for tid in cand.get(\"track_ids\", []) if cand.get', b'for tid in cand.get(\"track_ids\", [])]) if cand.get')

with open('tools/benchmark_sa_full_sweep.py', 'wb') as f:
    f.write(content)

print('Fixed bracket successfully!')
