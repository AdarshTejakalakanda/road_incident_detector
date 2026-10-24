import json

with open("artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json") as f:
    eval_audit = json.load(f)

with open("artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json") as f:
    supp_audit = json.load(f)

supp_ep_ids = set(e["episode_id"] for e in supp_audit)
quarantine_frames = [37, 64, 147, 180, 205, 328, 433, 471, 474, 483, 532, 570, 595, 681, 711, 720]

print("Matches by episode_id in supp_audit:")
by_id = [ep for ep in eval_audit if ep.get("episode_id") in supp_ep_ids]
print(f"Count by ep_id: {len(by_id)}")
for ep in by_id:
    print(f"  {ep.get('episode_id')}: frame={ep.get('anchor_frame')}, disp={ep.get('disposition')}, score={ep.get('score')}")

print("\nMatches by frame in quarantine_frames:")
by_frame = [ep for ep in eval_audit if ep.get("anchor_frame") in quarantine_frames]
print(f"Count by frame: {len(by_frame)}")
for ep in by_frame:
    print(f"  {ep.get('episode_id')}: frame={ep.get('anchor_frame')}, disp={ep.get('disposition')}, score={ep.get('score')}")

diff = [ep for ep in by_frame if ep.get("episode_id") not in supp_ep_ids]
print(f"\nEpisodes with frame in quarantine_frames but NOT in supp_audit: {len(diff)}")
for ep in diff:
    print(f"  {ep.get('episode_id')}: frame={ep.get('anchor_frame')}, disp={ep.get('disposition')}, score={ep.get('score')}")
