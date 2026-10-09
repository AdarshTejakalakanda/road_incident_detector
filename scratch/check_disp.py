import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from modules.vocabulary_gate import VocabularyGate

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\nAXTthLfgtI_00.mp4"
cap = cv2.VideoCapture(vpath)
w, h = 960, 720
road_poly = np.array(VocabularyGate().get_default_roadway_polygon((h, w)), dtype=np.int32)
mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(mask, [road_poly], 255)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, prev = cap.read()
tracked = []
nid = 1

for f in range(1, 75):
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

    for roi in current_rois:
        rbox = roi[:4]
        rcx, rcy = (rbox[0] + rbox[2]) / 2, (rbox[1] + rbox[3]) / 2
        best_match = None
        best_dist = 80.0
        for tc in tracked:
            tcx, tcy = (tc["box"][0] + tc["box"][2]) / 2, (tc["box"][1] + tc["box"][3]) / 2
            d = np.hypot(rcx - tcx, rcy - tcy)
            if d < best_dist:
                best_dist = d
                best_match = tc
        if best_match:
            tcx, tcy = (best_match["box"][0] + best_match["box"][2]) / 2, (best_match["box"][1] + best_match["box"][3]) / 2
            disp = np.hypot(rcx - tcx, rcy - tcy)
            best_match["total_disp"] = best_match.get("total_disp", 0.0) + disp
            best_match["box"] = rbox
            best_match["age"] += 1
            best_match["last_frame"] = f
            best_match["area"] = roi[4]
        else:
            tracked.append({"id": nid, "box": rbox, "age": 1, "last_frame": f, "area": roi[4], "total_disp": 0.0})
            nid += 1

    tracked = [tc for tc in tracked if f - tc["last_frame"] <= 2]
    if f in (54, 56, 58, 60):
        print(f"F{f}:")
        for tc in tracked:
            if tc["age"] >= 4:
                avg_speed = tc.get("total_disp", 0.0) / tc["age"]
                cid = tc["id"]
                cage = tc["age"]
                carea = tc["area"]
                cbox = tc["box"]
                print(f"  Cluster {cid}: age={cage}, area={carea:.0f}, speed={avg_speed:.2f} px/f, box={cbox}")
    prev = curr
cap.release()
