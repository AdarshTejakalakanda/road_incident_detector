import json

with open("telemetry_phase1_final_validation.json") as f:
    d = json.load(f)

for v in d["all_videos"]:
    print(f"{v['split'][:7]} {v['video_id']:15s} {v['status']:2s} cand={str(v['cand_frame']):>4s}")
