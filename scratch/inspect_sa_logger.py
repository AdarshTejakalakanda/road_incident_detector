with open('tools/benchmark_sa_full_sweep.py') as f:
    text = f.read()

if 'artifacts/sa_full_benchmark_sweep.json' in text:
    print('sa_full export found')
if 'post_ab_full' in text:
    print('post_ab_full export found')
