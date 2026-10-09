import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
from ultralytics import YOLO

full_path = os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", "nAXTthLfgtI_00.mp4")
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(full_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)

print("Testing nAXTthLfgtI_00 at imgsz=1280, conf=0.10:")
for f in range(50, 75):
    ret, frame = cap.read()
    if not ret:
        break
    p = model.predict(source=frame, conf=0.10, imgsz=1280, device="cpu", verbose=False)[0]
    if p.boxes is not None and len(p.boxes) > 0:
        c = p.boxes.conf.cpu().numpy()
        cl = p.boxes.cls.cpu().numpy().astype(int)
        names = [model.names[x] for x in cl]
        print(f"f={f}: {list(zip(names, [round(float(x), 2) for x in c]))}")
    else:
        print(f"f={f}: 0 boxes")
cap.release()
