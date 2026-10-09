import os, sys, pickle
sys.path.insert(0, '.')

with open('cache_videos_split_in_distribution_15.pkl', 'rb') as f:
    data = pickle.load(f)


for vid in data:
    v_id = vid.get('video_id')
    all_cands = []
    for f in vid.get('frames', []):
        cands = f[4]
        if cands:
            all_cands.extend(cands)
    if all_cands:
        c0 = all_cands[0]
        print(v_id, 'total cands:', len(all_cands), 'type:', type(c0), 'dir:', [k for k in dir(c0) if not k.startswith('_')])
        break
