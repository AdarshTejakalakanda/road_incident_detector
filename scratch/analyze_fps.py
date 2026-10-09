import json

with open("telemetry_phase1_final_validation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

vids = data["all_videos"]
fp_vids = [v for v in vids if v["fp_episodes"] > 0]
print(f"Total videos with FP episodes: {len(fp_vids)}")
total_fps = sum(v["fp_episodes"] for v in vids)
print(f"Total FP episodes: {total_fps}")

for v in fp_vids:
    print(f"{v['video_id']} ({v['split']}): status={v['status']}, gt={v['gt_frame']}, FP_episodes={v['fp_episodes']}, breakdown={v['fp_breakdown']}")
    for c in v['candidates']:
        diff = c['f'] - v['gt_frame']
        if abs(diff) > 35:
            print(f"    f={c['f']} (diff={diff:+d}) det={c['det']} tids={c['tids']}")
