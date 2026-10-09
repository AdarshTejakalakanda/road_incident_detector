import json

with open("scratch/z4_deep_records.json") as f:
    deep = json.load(f)

print("=" * 80)
print("PART 1 & 2: TRACK CONTINUITY & RAW DETECTIONS IN F150-F180")
print("=" * 80)

# Track IDs active in F150-F180
all_tids = set()
for f, rec in deep.items():
    for t in rec["tracks"]:
        all_tids.add(t["id"])

print(f"Total unique track IDs active across F150-F180: {sorted(list(all_tids))}")

# Check track life span across F150-F180
track_spans = {}
for f in range(150, 181):
    f_str = str(f)
    if f_str in deep:
        for t in deep[f_str]["tracks"]:
            tid = t["id"]
            if tid not in track_spans:
                track_spans[tid] = {"first": f, "last": f, "labels": set(), "statuses": set(), "provenances": set()}
            track_spans[tid]["last"] = f
            track_spans[tid]["labels"].add(t["cls"])
            track_spans[tid]["statuses"].add(t["status"])
            track_spans[tid]["provenances"].add(t["provenance"])

print("\nTrack Spans within F150-F180:")
for tid in sorted(track_spans.keys()):
    sp = track_spans[tid]
    print(f"  Track {tid:2d}: F{sp['first']} -> F{sp['last']} (span={sp['last'] - sp['first'] + 1}f) | labels={sp['labels']} | provs={sp['provenances']}")

print("\n" + "=" * 80)
print("FRAME-BY-FRAME DETAIL AROUND F165-F175")
print("=" * 80)
for f in range(165, 176):
    f_str = str(f)
    if f_str not in deep:
        continue
    rec = deep[f_str]
    raw_y = rec["raw_yolo"]
    mot_r = rec["motion_recovered"]
    adm_q = rec["admitted_quality"]
    trks = rec["tracks"]
    cands = rec["candidates"]
    supp = rec["suppressed"]

    print(f"\n--- FRAME {f} ---")
    print(f"  Raw YOLO ({len(raw_y)}): " + ", ".join([f"{d['cls']}(c={d['conf']:.2f}, box={d['bbox']})" for d in raw_y]))
    if mot_r:
        print(f"  Motion Recovered ({len(mot_r)}): " + ", ".join([f"{d['cls']}(c={d['conf']:.2f}, box={d['bbox']})" for d in mot_r]))
    print(f"  Admitted Quality Gate ({len(adm_q)}): " + ", ".join([f"{d['cls']}(c={d['conf']:.2f}, trust={d['trust']:.2f})" for d in adm_q]))
    
    # Relevant tracks near the right side / highway flow (x > 900)
    rel_tracks = [t for t in trks if t['id'] in [7, 10, 11, 14, 17, 18, 19, 20, 21, 22]]
    print(f"  Relevant Tracks ({len(rel_tracks)}):")
    for t in rel_tracks:
        print(f"    ID {t['id']:2d} [{t['cls']}]: box={t['bbox']}, spd={t['spd']:.1f} km/h, hdg={t['hdg']:.1f} deg, acc={t['accel']:.1f} m/s2, prov={t['provenance']}, age={t['age']}")

    if cands:
        print(f"  CANDIDATES TRIGGERED ({len(cands)}):")
        for c in cands:
            print(f"    Type={c['type']}, tracks={c['tracks']}, conf={c['conf']}")
    else:
        print("  CANDIDATES TRIGGERED: None")

    if supp:
        print(f"  SUPPRESSED IN ENGINE ({len(supp)}):")
        for s in supp:
            print(f"    Tracks={s['track_ids']}, Reason={s['reason']}, dist/metric={s['metric_val']}")
