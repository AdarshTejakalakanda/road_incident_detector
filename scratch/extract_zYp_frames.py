import cv2, os

video_path = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\zYp-honyJC0_00.mp4"
out_dir = r"scratch\zYp_frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

sample_frames = [1, 30, 60, 90, 120, 150, 180, 200, 220]
cur = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cur += 1
    if cur in sample_frames:
        out_path = os.path.join(out_dir, f"frame_{cur:03d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")
cap.release()
print("Done extracting sample frames.")
