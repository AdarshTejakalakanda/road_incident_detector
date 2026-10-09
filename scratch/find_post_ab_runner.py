import os

for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.py'):
            p=os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as file:
                    t = file.read()
                    if 'post_ab_full_benchmark' in t:
                        print('Found in:', p)
            except:
                pass
