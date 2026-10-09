import os, sys, pickle, glob
sys.path.insert(0, '.')

for filename in ['cache_videos_split_in_distribution_15.pkl', 'cache_videos_split_geo_aware_15.pkl']:
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(filename, 'videos:', len(data))
    for vid in data[:3]:
        v_id = vid.get('video_id')
        frames = vid.get('frames', [])
        print('  ', v_id, 'frames:', len(frames), 'gt_frame:', vid.get('gt_frame'), 'frame 0 type:', type(frames[0]), 'len:', len(frames[0]))
