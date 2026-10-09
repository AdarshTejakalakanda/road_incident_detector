import sys, os
sys.path.insert(0, os.path.abspath("."))
import pickle
from scripts.run_phase1_final_validation import Phase1ProductionIncidentPipeline

with open("cache_videos_iter_b_split_geo_aware_15.pkl", "rb") as f:
    vids = pickle.load(f)

v = [x for x in vids if x["video_id"] == "maJsgJsapH4_00"][0]

p = Phase1ProductionIncidentPipeline()
cands = []
for f_idx, pts, tr, kin, conv in v["frames"]:
    cands.extend(p.evaluate_frame(f_idx, pts, tr, kin, conv))

print("maJsgJsapH4_00 cands in Phase1ProductionIncidentPipeline:", len(cands), cands)
