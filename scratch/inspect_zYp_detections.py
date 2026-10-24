import cv2, os
import numpy as np

# Let's inspect the sample frames by looking at image statistics, vehicle detections and locations
from ultralytics import YOLO
model = YOLO('yolov8x.pt')

for f_name in sorted(os.listdir('scratch/zYp_frames')):
    if not f_name.endswith('.jpg'): continue
    path = os.path.join('scratch/zYp_frames', f_name)
    img = cv2.imread(path)
    res = model.predict(source=img, conf=0.15, device='cpu', verbose=False)[0]
    boxes = []
    if res.boxes is not None:
        for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            name = model.names[int(cl)]
            boxes.append(f"{name}@{c:.2f}({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})")
    print(f"{f_name}: {len(boxes)} dets -> {', '.join(boxes[:6])}")
