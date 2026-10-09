import sys, os
sys.path.insert(0, os.path.abspath('.'))
import pickle
from scratch.test_hierarchy_benchmark import RefinedProductionPipeline
from scripts.benchmark_iteration_b_detector_recovery import cluster_episodes

for split in ["in_distribution", "geo_aware"]:
    with open(f"cache_videos_iter_b_split_{split}_15.pkl", "rb") as f:
        vids = pickle.load(f)
    print(f"\n=== Remaining FPs in {split} ===")
    p = RefinedProductionPipeline(is_camera_calibrated=False)
    for v in vids:
        p.reset()
        raw = []
        for fr in v['frames']:
            c = p.evaluate_frame(fr[0], fr[1], fr[2], fr[3], fr[4])
            if c: raw.extend(c)
        eps = cluster_episodes(raw, max_gap_frames=25)
        for ep in eps:
            diff = ep['trigger_frame'] - v['gt_frame']
            if abs(diff) > 35:
                rep = ep['representative']
                print(f"  {v['video_id']:15s} @ f={ep['trigger_frame']:3d} (GT={v['gt_frame']:3d}, diff={diff:+4d}) det={rep.get('detector')} tids={rep.get('track_ids')}")
