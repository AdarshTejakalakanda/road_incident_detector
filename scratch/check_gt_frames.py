import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4")

# Let's save frames 120, 125, 128, 130, 140, 150, 160, 165, 170, 175
frames_to_save = [120, 125, 128, 130, 140, 150, 160, 165, 169, 170, 175]
saved = {}

f_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    if f_idx in frames_to_save:
        saved[f_idx] = frame

cap.release()

# Let's print out pixel statistics around GT bbox [906, 195, 978, 272] (on 1280x720) or native (1584x1080)
# GT normalized: x1=0.708, y1=0.272, x2=0.764, y2=0.377
h, w, _ = saved[170].shape
gx1, gy1, gx2, gy2 = int(0.708 * w), int(0.272 * h), int(0.764 * w), int(0.377 * h)
print(f"Video native resolution: {w}x{h}")
print(f"GT box in native coordinates: [{gx1}, {gy1}, {gx2}, {gy2}] (w={gx2-gx1}, h={gy2-gy1})")

for f in frames_to_save:
    img = saved[f]
    crop = img[gy1:gy2, gx1:gx2]
    mean_val = float(np.mean(crop))
    std_val = float(np.std(crop))
    print(f"Frame {f:3d}: GT crop mean={mean_val:.1f}, std={std_val:.1f}")
