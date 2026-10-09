with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('cand.get("track_ids", [*])', 'cand.get("track_ids", [])')
text = text.replace('fp_cand.get("track_ids", [*%u)', 'fp_cand.get("track_ids", [])')

with open('tools/benchmark_sa_full_sweep.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Fixed syntax error!')
