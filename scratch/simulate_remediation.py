import json

with open(" artifacts/post_ab_full_benchmark_sweep.json\, \r\, encoding=\utf-8\) as f:
 sweep = json.load(f)

# Let us test the canonical multi-vehicle bypass on the 71 FPs and 18 TPs
fps = json.load(open(\scratch/parsed_71_fps.json\, \r\, encoding=\utf-8\))
pvr = sweep[\per_video_results\]
tps = [x for x in pvr if x[\is_tp\]]

print(\Loaded\, len(fps), \FPs and\, len(tps), \TPs\)