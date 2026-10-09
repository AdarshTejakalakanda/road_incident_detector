with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    text = f.read()
idx = text.find('CANDIDATE_UNCONFIRMED')
print(text[idx-50:idx+800])
