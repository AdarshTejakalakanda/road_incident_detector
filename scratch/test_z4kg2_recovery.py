import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
from ultralytics import YOLO

full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", "Z4kg2Ev3vhk_00.mp4")
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(full_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
for f in range(150, 185):
    ret, frame = cap.read()
    if not ret:
        break
    p = model.predict(source=frame, conf=0.01, imgsz=640, device="cpu", verbose=False)[0]
    if p.boxes is not None and len(p.boxes) > 0:
        c = p.boxes.conf.cpu().numpy()
        cl = p.boxes.cls.cpu().numpy().astype(int)
        names = [model.names[x] for x in cl]
        print(f"f={f}: {list(zip(names, [round(x, 2) for x in c]))}")
    else:
        print(f"f={f}: 0 boxes")
cap.release()
