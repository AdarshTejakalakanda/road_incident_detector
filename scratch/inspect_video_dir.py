import os

video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"
meta_path = r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv"
print('video_dir exists:(, os.path.exists(video_dir))
print('meta_path exists:(, os.path.exists(meta_path))
if os.path.exists(video_dir):
    files = os.listdir(video_dir)
    print('videos count:', len(files))
