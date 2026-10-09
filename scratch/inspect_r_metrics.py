import json
r = json.load(open('artifacts/r_full_benchmark_sweep.json'))
print(json.dumps(r['metrics'], indent=2))
print('R split metrics:', json.dumps(r['split_metrics'], indent=2))
