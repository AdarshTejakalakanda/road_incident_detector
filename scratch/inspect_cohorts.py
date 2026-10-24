import pandas as pd, os

df = pd.read_csv(r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\metadata-real.csv")
video_dir = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos"

day_sample = df[df["day_time"] == "day"].sample(n=30, random_state=42)
night_sample = df[df["day_time"] == "night"].sample(n=30, random_state=42)

print("--- DAY SAMPLE (30) ---")
for idx, r in day_sample.iterrows():
    vid_id = os.path.basename(r["path"]).replace(".mp4", "")
    t = r["type"]
    f = r["accident_frame"]
    sec = r["accident_time"]
    tot = r["no_frames"]
    print(f"{vid_id:18s} | {t:10s} | f_gt={f:3d} ({sec:4.1f}s) | tot_f={tot:3d}")

print("\n--- NIGHT SAMPLE (30) ---")
for idx, r in night_sample.iterrows():
    vid_id = os.path.basename(r["path"]).replace(".mp4", "")
    t = r["type"]
    f = r["accident_frame"]
    sec = r["accident_time"]
    tot = r["no_frames"]
    print(f"{vid_id:18s} | {t:10s} | f_gt={f:3d} ({sec:4.1f}s) | tot_f={tot:3d}")
