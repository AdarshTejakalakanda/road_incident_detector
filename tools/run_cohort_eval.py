import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_pipeline import run_pipeline

def eval_single_video(task: dict) -> dict:
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
        
        # Evaluate TP / FP
        # A confirmed incident is TP if |frame - f_gt| <= 45
        tps = []
        fps = []
        for inc in confirmed:
            c = inc["candidate"]
            cf = c["frame"]
            if abs(cf - f_gt) <= 45 or (0 <= cf - f_gt <= 65):
                tps.append(inc)
            else:
                fps.append(inc)

        is_tp = len(tps) > 0
        is_fn = not is_tp
        fp_count = len(fps)

        first_tp_frame = tps[0]["candidate"]["frame"] if tps else None
        temporal_error = (first_tp_frame - f_gt) if first_tp_frame is not None else None

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
            "tp_frame": first_tp_frame,
            "temporal_error": temporal_error,
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
            "elapsed_s": round(time.time() - t0, 2),
            "error": str(e) + "\n" + traceback.format_exc(),
        }

if __name__ == "__main__":
    t0 = time.time()
    tasks = [
        {
            "video_id": "zYp-honyJC0_00",
            "video_path": r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\zYp-honyJC0_00.mp4",
            "accident_frame": 42,
            "accident_time": 2.125,
            "type": "single",
            "day_time": "day",
            "max_frames": 120,
        },
        {
            "video_id": "zYgBOSbkY3A_00",
            "video_path": r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\zYgBOSbkY3A_00.mp4",
            "accident_frame": 36,
            "accident_time": 1.625,
            "type": "t-bone",
            "day_time": "day",
            "max_frames": 120,
        }
    ]
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(eval_single_video, tasks))
    for r in results:
        print(f"[{r['video_id']}] TP={r['is_tp']}, FN={r['is_fn']}, FP={r['fp_count']}, elapsed={r['elapsed_s']}s")
    print(f"Total time: {time.time()-t0:.2f}s")
