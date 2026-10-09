with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()
idx = text.find('def passes_tier1_admission')
print(text[idx+1200:idx+2500])