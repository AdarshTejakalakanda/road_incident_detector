import json
import numpy as np

with open("scratch/night_diagnostic_results.json") as f:
    data = json.load(f)

for suite_name, suite_data in data.items():
    print(f"\n=======================================================")
    print(f"=== SUITE: {suite_name} ===")
    print(f"=======================================================")
    frame_data = suite_data["frame_data"]
    dc = frame_data["day_control"]
    print(f"Day control (Frame {dc['frame_idx']}): raw_lum={dc['raw_lum']:.2f}, enh_lum={dc['enh_lum']:.2f}, low_light_triggered={dc['is_low_light']}")
    print("Night frames luminance:")
    for nf in frame_data["night_frames"]:
        print(f"  Frame {nf['frame_idx']:3d}: raw_lum={nf['raw_lum']:.2f}, enh_lum={nf['enh_lum']:.2f}, low_light_triggered={nf['is_low_light']}")

    grid = suite_data["grid_results"]
    print("\nSummary Table (Total Counts across test frames):")
    print(f"{'Arm':<6} | {'Day/raw/0.25':<12} | {'Night/raw/0.10':<14} | {'Night/raw/0.25':<14} | {'Night/enh/0.10':<14} | {'Night/enh/0.25':<14} | {'Max Night Score':<15}")
    print("-" * 100)
    for arm, res in grid.items():
        day_raw_25 = res["day"]["raw_0.25"]["count"]

        n_raw_10 = sum(nf["raw_0.1"]["count"] for nf in res["night"])
        n_raw_25 = sum(nf["raw_0.25"]["count"] for nf in res["night"])
        n_enh_10 = sum(nf["enh_0.1"]["count"] for nf in res["night"])
        n_enh_25 = sum(nf["enh_0.25"]["count"] for nf in res["night"])

        all_night_scores = []
        for nf in res["night"]:
            all_night_scores.extend(nf["raw_0.1"]["scores"])
            all_night_scores.extend(nf["enh_0.1"]["scores"])
        max_score = max(all_night_scores) if all_night_scores else 0.0

        print(f"{arm:<6} | {day_raw_25:<12} | {n_raw_10:<14} | {n_raw_25:<14} | {n_enh_10:<14} | {n_enh_25:<14} | {max_score:<15.4f}")

    # Per-frame breakdown
    print("\nPer-Frame Breakdown:")
    for arm, res in grid.items():
        print(f"\n--- Arm {arm} ---")
        for nf in res["night"]:
            f_idx = nf["frame_idx"]
            c_r10 = nf["raw_0.1"]["count"]
            c_r25 = nf["raw_0.25"]["count"]
            c_e10 = nf["enh_0.1"]["count"]
            c_e25 = nf["enh_0.25"]["count"]
            s_max = max(nf["raw_0.1"]["scores"] + nf["enh_0.1"]["scores"]) if (nf["raw_0.1"]["scores"] or nf["enh_0.1"]["scores"]) else 0.0
            print(f"  Frame {f_idx:3d}: raw@0.10={c_r10:2d}, raw@0.25={c_r25:2d}, enh@0.10={c_e10:2d}, enh@0.25={c_e25:2d}, max_s={s_max:.3f}")

    # Detailed statistics for Task 4
    print("\nTask 4 Score Distribution & Root Cause Analysis:")
    for arm, res in grid.items():
        all_raw_10 = []
        for nf in res["night"]:
            all_raw_10.extend(nf["raw_0.1"]["scores"])
        
        all_raw_25 = []
        for nf in res["night"]:
            all_raw_25.extend(nf["raw_0.25"]["scores"])

        all_enh_10 = []
        for nf in res["night"]:
            all_enh_10.extend(nf["enh_0.1"]["scores"])

        all_enh_25 = []
        for nf in res["night"]:
            all_enh_25.extend(nf["enh_0.25"]["scores"])

        total_r10 = len(all_raw_10)
        total_r25 = len(all_raw_25)
        total_e10 = len(all_enh_10)
        total_e25 = len(all_enh_25)

        s_min = float(np.min(all_raw_10)) if total_r10 > 0 else 0.0
        s_max = float(np.max(all_raw_10)) if total_r10 > 0 else 0.0
        s_med = float(np.median(all_raw_10)) if total_r10 > 0 else 0.0

        # % detections near threshold (within 0.05 of 0.25, i.e., [0.20, 0.30] or [0.10, 0.15])
        near_tau25 = [s for s in all_raw_10 if 0.20 <= s <= 0.30]
        pct_near_tau25 = (len(near_tau25) / total_r10 * 100.0) if total_r10 > 0 else 0.0

        near_tau10 = [s for s in all_raw_10 if 0.10 <= s <= 0.15]
        pct_near_tau10 = (len(near_tau10) / total_r10 * 100.0) if total_r10 > 0 else 0.0

        print(f"\nArm {arm}:")
        print(f"  Count raw@0.10: {total_r10}, raw@0.25: {total_r25}, enh@0.10: {total_e10}, enh@0.25: {total_e25}")
        print(f"  Score Range: [{s_min:.3f}, {s_max:.3f}], Median: {s_med:.3f}")
        print(f"  Detections near tau=0.25 (0.20-0.30): {len(near_tau25)} ({pct_near_tau25:.1f}%)")
        print(f"  Detections near tau=0.10 (0.10-0.15): {len(near_tau10)} ({pct_near_tau10:.1f}%)")
