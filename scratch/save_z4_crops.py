import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\Z4kg2Ev3vhk_00.mp4")

# Extract frames 120 through 180 every 5 frames and around 165-175 every 1 frame
frames = {}
f = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    f += 1
    if (120 <= f <= 135) or (165 <= f <= 175):
        frames[f] = frame
cap.release()

print("Analyzing visual frames...")
# GT box on 1584x1080 is [1121, 293, 1210, 407]
# Let's save crops to disk in scratch/z4_crops
import os
os.makedirs("scratch/z4_crops", exist_ok=True)

for idx in [126, 128, 130, 168, 169, 170, 171, 172]:
    if idx in frames:
        cv2.imwrite(f"scratch/z4_crops/frame_{idx}_full.jpg", frames[idx])
        crop = frames[idx][250:450, 1050:1250]
        cv2.imwrite(f"scratch/z4_crops/frame_{idx}_roi.jpg", crop)
        print(f"Saved scratch/z4_crops/frame_{idx}_roi.jpg")
