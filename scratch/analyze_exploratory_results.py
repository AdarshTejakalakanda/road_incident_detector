import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

with open("scratch/exploratory_experiment_results.json") as f:
    results = f.read()
    data = json.loads(results)

with open("scratch/exploratory_30_manifest.json") as f:
    manifest = json.load(f)

print("Experiment Metadata:")
print(data["metadata"])

per_arm = data["per_arm_results"]

# Analysis by arm and cohort
# Cohorts: verified-night (12), mislabeled-night (6), day (12), all (30)

summary_rows = []

for arm_key, arm_data in per_arm.items():
    model_name = arm_data["model_name"]
    vids = arm_data["per_video"]
    
    # Stratified by cohort
    for cohort_filter in ["all", "verified-night", "mislabeled-night", "day"]:
        if cohort_filter == "all":
            subset = [v for v in vids if v.get("status") != "NOT_EVALUATED"]
        else:
            subset = [v for v in vids if v.get("cohort") == cohort_filter and v.get("status") != "NOT_EVALUATED"]
            
        if not subset:
            summary_rows.append({
                "arm": arm_key,
                "model": model_name,
                "cohort": cohort_filter,
                "n_vids": 0,
                "status": "NOT_EVALUATED",
            })
            continue

        n_vids = len(subset)
        total_frames = sum(v["frames_processed"] for v in subset)
        total_dur_s = sum(v["duration_s"] for v in subset)
        
        # Conf 0.25
        dets_25 = sum(v["conf_0_25"]["total_detections"] for v in subset)
        tp_25 = sum(v["conf_0_25"]["alerts"]["tp"] for v in subset)
        fp_25 = sum(v["conf_0_25"]["alerts"]["fp"] for v in subset)
        fn_25 = sum(v["conf_0_25"]["alerts"]["fn"] for v in subset)
        alerts_25 = sum(v["conf_0_25"]["alerts"]["alert_count"] for v in subset)
        # Pre-impact duration across subset for FA/hour
        pre_dur_hours = sum(max(0.0, (v["accident_frame"] - 35) / v["fps"]) / 3600.0 for v in subset)
        fa_rate_25 = (fp_25 / pre_dur_hours) if pre_dur_hours > 0 else 0.0
        recall_25 = (tp_25 / (tp_25 + fn_25)) if (tp_25 + fn_25) > 0 else 0.0

        # Conf 0.10
        dets_10 = sum(v["conf_0_10"]["total_detections"] for v in subset)
        tp_10 = sum(v["conf_0_10"]["alerts"]["tp"] for v in subset)
        fp_10 = sum(v["conf_0_10"]["alerts"]["fp"] for v in subset)
        fn_10 = sum(v["conf_0_10"]["alerts"]["fn"] for v in subset)
        alerts_10 = sum(v["conf_0_10"]["alerts"]["alert_count"] for v in subset)
        fa_rate_10 = (fp_10 / pre_dur_hours) if pre_dur_hours > 0 else 0.0
        recall_10 = (tp_10 / (tp_10 + fn_10)) if (tp_10 + fn_10) > 0 else 0.0

        # Latencies
        infer_means = [v["latency_ms"]["infer_mean"] for v in subset]
        infer_p50s = [v["latency_ms"]["infer_p50"] for v in subset]
        infer_p95s = [v["latency_ms"]["infer_p95"] for v in subset]
        track_means = [v["latency_ms"]["track_mean"] for v in subset]
        event_means = [v["latency_ms"]["event_mean"] for v in subset]
        decode_means = [v["latency_ms"]["decode_mean"] for v in subset]

        # Resources
        cpu_means = [v["resources"]["mean_cpu_pct"] for v in subset]
        ram_peaks = [v["resources"]["peak_ram_mb"] for v in subset]

        summary_rows.append({
            "arm": arm_key,
            "model": model_name,
            "cohort": cohort_filter,
            "n_vids": n_vids,
            "status": "EVALUATED",
            "frames": total_frames,
            # Conf 0.25 metrics
            "dets_25": dets_25,
            "dets_per_s_25": round(dets_25 / total_dur_s, 1) if total_dur_s > 0 else 0.0,
            "tp_25": tp_25,
            "fp_25": fp_25,
            "fn_25": fn_25,
            "recall_25": round(recall_25 * 100, 1),
            "fa_hr_25": round(fa_rate_25, 1),
            # Conf 0.10 metrics
            "dets_10": dets_10,
            "dets_per_s_10": round(dets_10 / total_dur_s, 1) if total_dur_s > 0 else 0.0,
            "tp_10": tp_10,
            "fp_10": fp_10,
            "fn_10": fn_10,
            "recall_10": round(recall_10 * 100, 1),
            "fa_hr_10": round(fa_rate_10, 1),
            # Latency (ms)
            "lat_infer_mean": round(float(np.mean(infer_means)), 1),
            "lat_infer_p50": round(float(np.median(infer_p50s)), 1),
            "lat_infer_p95": round(float(np.mean(infer_p95s)), 1),
            "lat_track_mean": round(float(np.mean(track_means)), 1),
            "lat_event_mean": round(float(np.mean(event_means)), 1),
            "lat_decode_mean": round(float(np.mean(decode_means)), 1),
            # Resources
            "cpu_mean_pct": round(float(np.mean(cpu_means)), 1),
            "ram_peak_mb": round(float(np.max(ram_peaks)), 1),
        })

df_sum = pd.DataFrame(summary_rows)
print("\n=== SUMMARY TABLE: ALL COHORTS AGGREGATE ===")
print(df_sum[df_sum["cohort"] == "all"][["arm", "dets_25", "dets_10", "tp_25", "fp_25", "fn_25", "recall_25", "fa_hr_25", "tp_10", "fp_10", "recall_10", "fa_hr_10", "lat_infer_mean", "lat_infer_p95", "ram_peak_mb"]])

print("\n=== SUMMARY TABLE: PRIMARY COHORT (VERIFIED-NIGHT) ===")
print(df_sum[df_sum["cohort"] == "verified-night"][["arm", "dets_25", "dets_10", "tp_25", "fp_25", "recall_25", "fa_hr_25", "tp_10", "fp_10", "recall_10", "fa_hr_10", "lat_infer_mean", "ram_peak_mb"]])

print("\n=== SUMMARY TABLE: SECONDARY COHORT (MISLABELED-NIGHT) ===")
print(df_sum[df_sum["cohort"] == "mislabeled-night"][["arm", "dets_25", "dets_10", "tp_25", "fp_25", "recall_25", "fa_hr_25", "tp_10", "fp_10", "recall_10", "fa_hr_10"]])

print("\n=== SUMMARY TABLE: CONTROL COHORT (DAYTIME) ===")
print(df_sum[df_sum["cohort"] == "day"][["arm", "dets_25", "dets_10", "tp_25", "fp_25", "recall_25", "fa_hr_25", "tp_10", "fp_10", "recall_10", "fa_hr_10"]])

# Class breakdown analysis per arm
print("\n=== CLASS BREAKDOWN SUMMARY (CONF 0.25) ===")
for arm_key, arm_data in per_arm.items():
    cls_counter = Counter()
    for v in arm_data["per_video"]:
        if v.get("status") != "NOT_EVALUATED":
            cls_counter.update(v["conf_0_25"]["class_counts"])
    print(f"{arm_key} ({arm_data['model_name']}): {dict(cls_counter)}")
