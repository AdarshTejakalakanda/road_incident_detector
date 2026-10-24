from run_pipeline import run_pipeline
import json

# Let's run a diagnostic on track 52 around frame 245
# We can load the candidate from 5uh8-3q51uI_00_ep_03_f245_barrier_strike.json
with open('output/roi_crops/5uh8-3q51uI_00_ep_03_f245_barrier_strike.json') as f:
    d = json.load(f)

print("Candidate:")
print(json.dumps(d, indent=2))
