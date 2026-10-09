with open('tools/benchmark_sa_full_sweep.py', 'r', encoding='utf-8') as f:
    text = f.read()
idx = text.find('fp_confirmed_count +=')
print(text[idx-200:idx+400])
