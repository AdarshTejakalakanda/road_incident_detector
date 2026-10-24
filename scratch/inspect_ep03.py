import sys, os
sys.path.insert(0, os.path.abspath('.'))
import json

# Let's inspect the sidecar for ep03 f245
with open('output/roi_crops/5uh8-3q51uI_00_ep_03_f245_barrier_strike.json') as f:
    meta = json.load(f)

print("Episode JSON:")
print(json.dumps(meta, indent=2))
