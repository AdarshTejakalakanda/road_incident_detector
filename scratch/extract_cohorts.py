import json, os, sys
sys.path.insert(0, '.')
from modules.vlm_arbitrator import passes_tier1_admission, GroqArbitrator

with open('artifacts/post_ab_full_benchmark_sweep.json') as f:
    post = json.load(f)


arbiter = GroqArbitrator(mock_fallback=True)

tp_verified = []
fn_verified = []
fp_daytime = []

for r in post["per_video_results"]:
    vid = r["video_id"]
    split = r["split_type"]
    gt = r["gt_frame"]
    cls = r["classification"]
    fp_cnt = r["fp_confirmed_count"]
    day_t = r.get("day_time", "day")

    if cls == "TP":
        mc = r["et_tp_result"] # vlm_tp_result
        vres = r.get("vlm_tp_result", {})
        tp_verified.append({
            "video_id": vid,
            "split": split,
            "gt_frame": gt,
            "source": vres.get("source"),
            "verdict": vres.get("verdict"),
            "is_collision": vres.get("is_collision"),
            "bypass_reason": vres.get("bypass_telemetry", {}).get("bypass_reason"),
        })
    elif cls == "FNG" or cls == "FN" or r.get("is_fn"):
        fn_verified.append({
            "video_id": vid,
            "split": split,
            "gt_frame": gt,
            "failure_gate": r.get("failure_gate"),
            "raw_cands_count": r.get("raw_candidates_count", 0),
            "in_window_episodes": r.get("in_window_episodes_count", 0),
            "vlm_calls": r.get("vlm_calls", 0),
        })

    if fp_cnt > 0:
        fp_daytime.append({
            "video_id": vid,
            "split": split,
            "day_time": day_t,
            "fp_confirmed_count": fp_cnt,
            "out_of_window_episodes": r.get("out_of_window_episodes_count", 0),
        })

print("Cohort C (TPs):", len(tp_verified))
print("Cohort D (FNs):", len(fn_verified))
print("Cohort A (Daytime FPs):", len(fp_daytime), "sum fp_confirmed:", sum(x["fp_confirmed_count"] for x in fp_daytime))
