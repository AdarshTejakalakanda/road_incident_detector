import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle
from scratch.test_hierarchy_benchmark import RefinedProductionPipeline

with open("cache_videos_iter_b_split_in_distribution_15.pkl", "rb") as f:
    vids = pickle.load(f)

v = [x for x in vids if x["video_id"] == "LiM5LPSZoow_00"][0]

p = RefinedProductionPipeline()
for f_idx, pts, tr, kin, conv in v["frames"]:
    if f_idx <= 20:
        cands = p.evaluate_frame(f_idx, pts, tr, kin, conv)
        if cands:
            print(f"Frame {f_idx}: {cands}")
