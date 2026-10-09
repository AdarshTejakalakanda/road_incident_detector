import base64

patch_py = ''''
# Stage T Arbitrator Patcher
with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

print('Read len:', len(text))
''''

exec(patch_py)
