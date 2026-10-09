import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from modules.vocabulary_gate import VocabularyGate
from ultralytics import YOLO

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
cap = cv2.VideoCapture(vpath)
w, h = 960, 720
road_poly = np.array(VocabularyGate().get_default_roadway_polygon((h, w)), dtype=np.int32)
mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(mask, [road_poly], 255)

model = YOLO("yolo26n.pt")
cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
ret, prev = cap.read()
tracked = []
nid = 1

def compute_box_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

for f in range(46, 75):
    ret, curr = cap.read()
    if not ret:
        break
    diff = cv2.absdiff(curr, prev)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=mask)
    _, thresh = cv2.threshold(masked_diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_rois = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if 250 <= area <= 3500:
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad_x, pad_y = int(bw * 0.5), int(bh * 0.5)
            current_rois.append((max(0, x - pad_x), max(0, y - pad_y), min(w, x + bw + pad_x), min(h, y + bh + pad_y), area))

    matched_ids = set()
    for roi in current_rois:
        rbox = roi[:4]
        rcx, rcy = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2
        best_match = None
        best_dist = 80.0
        for tc in tracked:
            if tc["id"] in matched_ids:
                continue
            tcx, tcy = (tc["box"][0] + tc["box"][2]) / 2, (tc["box"][1] + tc["box"][3]) / 2
            d = np.hypot(rcx - tcx, rcy - tcy)
            if d < best_dist:
                best_dist = d
                best_match = tc
        if best_match:
            best_match["box"] = rbox
            best_match["age"] += 1
            best_match["last_frame"] = f
            best_match["area"] = roi[4]
            matched_ids.add(best_match["id"])
        else:
            tracked.append({"id": nid, "box": rbox, "age": 1, "last_frame": f, "area": roi[4], "active_hits": 0})
            nid += 1

    tracked = [tc for tc in tracked if f - tc["last_frame"] <= 2]

    # Primary pass
    yres = model.predict(source=curr, imgsz=640, conf=0.02, verbose=False)[0]
    p_veh = []
    if yres.boxes is not None and len(yres.boxes) > 0:
        for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy()):
            if model.names[int(cl)].lower() in ("car", "truck", "bus", "motorcycle", "van", "train"):
                p_veh.append(b)

    cands = [tc for tc in tracked if tc["age"] >= 4 and not any(compute_box_iou(tc["box"], pb) > 0.05 for pb in p_veh)]
    if cands:
        # Prioritize active hits, then area
        best = max(cands, key=lambda c: (c.get("active_hits", 0) > 0, c["area"]))
        bx1, by1, bx2, by2 = best["box"]
        crop = curr[by1:by2, bx1:bx2]
        crop_320 = cv2.resize(crop, (320, 320))
        zres = model.predict(source=crop_320, imgsz=320, conf=0.05, verbose=False)[0]
        dets = []
        if zres.boxes is not None and len(zres.boxes) > 0:
            for zb, zc, zcl in zip(zres.boxes.xyxy.cpu().numpy(), zres.boxes.conf.cpu().numpy(), zres.boxes.cls.cpu().numpy()):
                cname = model.names[int(zcl)].lower()
                if cname == "train":
                    cname = "truck"
                if cname in ("car", "truck", "bus", "motorcycle"):
                    sx = (bx2 - bx1) / 320.0
                    sy = (by2 - by1) / 320.0
                    gx1 = int(bx1 + zb[0] * sx)
                    gy1 = int(by1 + zb[1] * sy)
                    gx2 = int(bx1 + zb[2] * sx)
                    gy2 = int(by1 + zb[3] * sy)
                    dets.append(f"{cname}:{zc:.2f}:[{gx1},{gy1},{gx2},{gy2}]")
        if dets:
            best["active_hits"] = best.get("active_hits", 0) + 1
        else:
            best["active_hits"] = max(0, best.get("active_hits", 0) - 1)
        cid = best["id"]
        cage = best["age"]
        print(f"F{f}: Cluster {cid} (age={cage}, hits={best.get('active_hits',0)}) -> {dets}")
    prev = curr
cap.release()
