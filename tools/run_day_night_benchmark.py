import os
import sys
import time
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_pipeline import run_pipeline

def evaluate_single_video_task(task: dict) -> dict:
    import torch
    torch.set_num_threads(4)

    vpath = task["video_path"]
    vid_id = task["video_id"]
    f_gt = task["accident_frame"]
    t_gt = task["accident_time"]
    vtype = task["type"]
    day_time = task["day_time"]
    max_f = task["max_frames"]

    t0 = time.time()
    try:
        res = run_pipeline(
            video_path=vpath,
            max_frames=max_f,
            save_video=False,
            save_evidence=True,
        )
        elapsed = time.time() - t0
        confirmed = res.get("confirmed_incidents", [])
        
        # Temporal matching tolerance: [-45 frames, +65 frames]
        tps = []
        fps = []
        for inc in confirmed:
            c = inc["candidate"]
            cf = c["frame"]
            if -45 <= (cf - f_gt) <= 65:
                tps.append(inc)
            else:
                fps.append(inc)

        is_tp = len(tps) > 0
        is_fn = not is_tp
        fp_count = len(fps)

        first_tp = tps[0] if tps else None
        tp_frame = first_tp["candidate"]["frame"] if first_tp else None
        temporal_error = (tp_frame - f_gt) if tp_frame is not None else None
        crash_type = first_tp["arbitration"].get("collision_type", first_tp["candidate"].get("event_type", "unknown")) if first_tp else None
        conf = first_tp["arbitration"].get("confidence", 0.0) if first_tp else 0.0

        return {
            "video_id": vid_id,
            "day_time": day_time,
            "type": vtype,
            "accident_frame": f_gt,
            "accident_time": t_gt,
            "frames_processed": res.get("frames_processed", 0),
            "candidates_count": res.get("candidates_count", 0),
            "confirmed_count": len(confirmed),
            "is_tp": is_tp,
            "is_fn": is_fn,
            "fp_count": fp_count,
            "tp_frame": tp_frame,
            "temporal_error": temporal_error,
            "crash_type": crash_type,
            "confidence": conf,
            "elapsed_s": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        import traceback
        return {
            "video_id": vid_id,
            "day_time": day_time,
            "type": vtype,
            "accident_frame": f_gt,
            "accident_time": t_gt,
            "frames_processed": 0,
            "candidates_count": 0,
            "confirmed_count": 0,
            "is_tp": False,
            "is_fn": True,
            "fp_count": 0,
            "tp_frame": None,
            "temporal_error": None,
            "crash_type": None,
            "confidence": 0.0,
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(e) + "\n" + traceback.format_exc(),
        }

def run_cohort_benchmark(cohort_name: str, sample_df: pd.DataFrame, video_dir: str, max_workers: int = 3) -> dict:
    print("\n" + "=" * 90)
    print(f"STARTING BENCHMARK RUN: {cohort_name.upper()} COHORT (30 VIDEOS)")
    print("=" * 90)
    
    tasks = []
    for _, r in sample_df.iterrows():
        vid_id = os.path.basename(r["path"]).replace(".mp4", "")
        vpath = os.path.join(r"C:\Users\Adarsh Teja Kalakand\Downloads\archive", r["path"])
        f_gt = int(r["accident_frame"])
        t_gt = float(r["accident_time"])
        tot_f = int(r["no_frames"])
        max_f = min(tot_f, max(140, f_gt + 65))

        tasks.append({
            "video_id": vid_id,
            "video_path": vpath,
            "accident_frame": f_gt,
            "accident_time": t_gt,
            "type": r["type"],
            "day_time": r["day_time"],
            "max_frames": max_f,
        })

    results = []
    t_start = time.time()
    completed_count = 0
    total_tasks = len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single_video_task, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed_count += 1
            status = "TP" if res["is_tp"] else "FN"
            fp_str = f"FP={res['fp_count']}" if res['fp_count'] > 0 else "FP=0"
            err_str = f" | ERR: {res['temporal_error']:+d}f" if res['temporal_error'] is not None else ""
            print(f"[{completed_count:02d}/{total_tasks:02d}] {res['video_id']:18s} | {res['type']:10s} | {status:2s} ({fp_str}{err_str}) | {res['elapsed_s']:4.1f}s")

    t_total = time.time() - t_start

    # Compute Aggregate Metrics
    tp_count = sum(1 for r in results if r["is_tp"])
    fn_count = sum(1 for r in results if r["is_fn"])
    fp_count = sum(r["fp_count"] for r in results)

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Stratified by incident type
    types = sorted(list(set(r["type"] for r in results)))
    type_breakdown = {}
    for t in types:
        sub = [r for r in results if r["type"] == t]
        sub_tp = sum(1 for r in sub if r["is_tp"])
        sub_fn = sum(1 for r in sub if r["is_fn"])
        sub_fp = sum(r["fp_count"] for r in sub)
        sub_prec = sub_tp / (sub_tp + sub_fp) if (sub_tp + sub_fp) > 0 else 0.0
        sub_rec = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else 0.0
        sub_f1 = (2 * sub_prec * sub_rec) / (sub_prec + sub_rec) if (sub_prec + sub_rec) > 0 else 0.0
        type_breakdown[t] = {
            "total": len(sub),
            "tp": sub_tp,
            "fn": sub_fn,
            "fp": sub_fp,
            "precision": round(sub_prec, 4),
            "recall": round(sub_rec, 4),
            "f1": round(sub_f1, 4),
        }

    summary = {
        "cohort": cohort_name,
        "total_videos": total_tasks,
        "tp_count": tp_count,
        "fn_count": fn_count,
        "fp_count": fp_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "total_time_s": round(t_total, 2),
        "average_time_per_video_s": round(t_total / total_tasks, 2),
        "type_breakdown": type_breakdown,
        "video_results": results,
    }

    print("\n" + "=" * 90)
    print(f"BENCHMARK SUMMARY: {cohort_name.upper()} COHORT")
    print("=" * 90)
    print(f"Total Videos:      {total_tasks}")
    print(f"True Positives:    {tp_count} / {total_tasks}")
    print(f"False Negatives:   {fn_count} / {total_tasks}")
    print(f"False Positives:   {fp_count}")
    print(f"PRECISION:         {precision * 100:.1f}%")
    print(f"RECALL:            {recall * 100:.1f}%")
    print(f"F1 SCORE:          {f1:.3f}")
    print(f"Total Execution:   {t_total:.1f}s ({t_total/60:.1f} min)")
    print("-" * 90)
    print("STRATIFIED BY INCIDENT TYPE:")
    for t, b in type_breakdown.items():
        print(f"  * {t:12s} ({b['total']:2d} vids): Recall = {b['recall']*100:5.1f}% ({b['tp']}/{b['total']}) | Precision = {b['precision']*100:5.1f}% | FP = {b['fp']}")
    print("=" * 90)

    return summary

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\metadata-real.csv")
    video_dir = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos"

    day_sample = df[df["day_time"] == "day"].sample(n=30, random_state=42)
    night_sample = df[df["day_time"] == "night"].sample(n=30, random_state=42)

    # 1. Run Day Cohort
    day_summary = run_cohort_benchmark("day", day_sample, video_dir, max_workers=3)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/benchmark_day_30.json", "w", encoding="utf-8") as f:
        json.dump(day_summary, f, indent=2)
    print("Saved day benchmark to: artifacts/benchmark_day_30.json")

    # 2. Run Night Cohort
    night_summary = run_cohort_benchmark("night", night_sample, video_dir, max_workers=3)
    with open("artifacts/benchmark_night_30.json", "w", encoding="utf-8") as f:
        json.dump(night_summary, f, indent=2)
    print("Saved night benchmark to: artifacts/benchmark_night_30.json")
