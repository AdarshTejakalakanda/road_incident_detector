import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4")

f_idx = 0
frames = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    if 120 <= f_idx <= 180:
        frames[f_idx] = frame
cap.release()

print(f"Captured {len(frames)} frames from F120 to F180.")

# Let's inspect optical flow or frame difference around GT box [1121, 293, 1210, 407]
# Let's see what is moving into or out of [1121, 293, 1210, 407]
diffs = []
for f in range(125, 179):
    gray1 = cv2.cvtColor(frames[f][250:450, 1050:1250], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frames[f+1][250:450, 1050:1250], cv2.COLOR_BGR2GRAY)
    diff = np.mean(cv2.absdiff(gray1, gray2))
    diffs.append((f, diff))

# Top 10 motion frames
top_diffs = sorted(diffs, key=lambda x: x[1], reverse=True)[:15]
print("Top frame-to-frame diffs in ROI [250:450, 1050:1250]:")
for f, d in top_diffs:
    print(f"  F{f}->F{f+1}: diff={d:.2f}")

# Check what vehicles are detected in native coordinates around F128 vs F170
