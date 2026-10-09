import os, sys, pickle
sys.path.insert(0, '.')

with open('cache_videos_split_in_distribution_15.pkl', 'rb') as f:
    data = pickle.load(f)


f0 = data[0]['frames'][0]
print('frame_id:', f0[0], 'ts_ms:', f0[1])
print('el 2: raw_dets keys:', list(f0[2].keys()))
print('el 3: tracks keys:', list(f0[3].keys()))
print('el 4: candidates len:', len(f0[4]))
if f0[4]:
    print('  cand 0 keys:', list(f0[4][0].keys()))
