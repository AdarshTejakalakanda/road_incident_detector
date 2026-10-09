import json
d = json.load(open('artifacts/post_ab_full_benchmark_sweep.json', encoding='utf-8'))
print(list(d.keys()))
