import os, sys, pickle
sys.path.insert(0, '.')

with open('cache_videos_split_in_distribution_15.pkl', 'rb') as f:
    data = pickle.load(f)

print('Data type:', type(data))
if isinstance(data, list):
    print('Len:', len(data))
    print('Item 0 type:', type(data[0]))
    if isinstance(data[0], dict):
        print('Item 0 keys:', list(data[0].keys()))
        print('item 0 video_id:', data[0].get('video_id'))
