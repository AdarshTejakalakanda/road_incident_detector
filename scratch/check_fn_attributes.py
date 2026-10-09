import pandas as pd

df = pd.read_csv(r"C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv")
fn_vids = [
    "82sQ7N-aHOw_00",
    "Z4kg2Ev3vhk_00",
    "UarP8qU1S-c_00",
    "UarP8qU1S-c_01",
    "VozrseCzhQc_00",
    "fzWY0vLAXzI_00",
    "kABWTDKgc9U_00",
    "nAXTthLfgtI_00",
]

for vid in fn_vids:
    m = df[df["path"].str.contains(vid)]
    if len(m) > 0:
        r = m.iloc[0]
        w = r["width"]
        h = r["height"]
        bx_w = (r["x2"] - r["x1"]) * w
        bx_h = (r["y2"] - r["y1"]) * h
        scaled_w = bx_w * 640 / w
        scaled_h = bx_h * 360 / h
        print(f"{vid} | type={r['type']} | res={w}x{h} | time={r['day_time']} | quality={r['quality']} | raw_box={bx_w:.1f}x{bx_h:.1f} | scaled_box={scaled_w:.1f}x{scaled_h:.1f}")
