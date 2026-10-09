import json

with open("artifacts/stage_t_remediated_5uh8_audit.json") as f:
    stage_t = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_16 = json.load(f)

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_53 = json.load(f)

fps_24 = [c for c in stage_t["all_candidates"] if c.get("event_type") == "collision_candidate" and c.get("frame") != 182]

print("24 FP frames from stage_t_remediated_5uh8_audit:")
fp_frames = [c["frame"] for c in fps_24]
print(fp_frames)

print("\n16 Quarantined frames from aftermath_suppression_audit:")
supp_frames = [ep["anchor_frame"] for ep in supp_16]
print(supp_frames)

print("\n53 Evaluated frames from aftermath_evaluation_audit:")
eval_frames = [ep["anchor_frame"] for ep in eval_53]
print(eval_frames)

# Check overlaps between 24 FPs and 16 quarantined
exact_match = set(fp_frames).intersection(set(supp_frames))
print(f"\nExact frame overlap between 24 FPs and 16 Quarantined: {exact_match}")

# Check near overlaps (+/- 5 frames)
near_matches = []
for f_fp in fp_frames:
    for f_sq in supp_frames:
        if abs(f_fp - f_sq) <= 5:
            near_matches.append((f_fp, f_sq))
print(f"Near frame matches (+/- 5 frames): {near_matches}")
