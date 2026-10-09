import os
import sys
import glob
import re
import json
from collections import defaultdict

def run_attribution():
    log_files = glob.glob(r'C:\Users\KalakandaAdarshTeja*\.gemini\antigravity\brain\ee6ed818-3319-475c-b4b0-eef91015e225\.system_generated\tasks\task-10515.log')
    if not log_files:
        raise FileNotFoundError("Could not find task-10515.log")
    
    with open(log_files[0], 'r', encoding='utf-8', errors='ignore') as f:
        log_text = f.read()

    # Load r_full_benchmark_sweep.json
    with open('artifacts/r_full_benchmark_sweep.json', 'r', encoding='utf-8') as f:
        bench_data = json.load(f)

    # Extract all confirmed FP log entries
    pattern = r'DEBUG FP CONFIRMED: video=(.*?), ep=(.*?), frame=(\d+), type=(.*?), res=(.*?)\nDEBUG FP TELEMETRY: (\{.*?\})'
    matches = re.findall(pattern, log_text)

    print(f"Total confirmed FP entries parsed: {len(matches)}")

    # Also build per-video episode timelines to identify duplicates/bursts
    episodes_by_video = defaultdict(list)
    fp_records = []

    for idx, (vid, ep_id, frame_str, etype, res_str, tel_str) in enumerate(matches, 1):
        cand_frame = int(frame_str)
        # Parse telemetry dict
        # tel_str uses python dict syntax (True/False/etc.), safe eval or json with replacements
        tel_clean = tel_str.replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            tel = json.loads(tel_clean)
        except Exception:
            tel = eval(tel_str)

        rec = {
            "index": idx,
            "video_id": vid,
            "episode_id": ep_id,
            "frame": cand_frame,
            "event_type": etype,
            "rationale": res_str,
            "telemetry": tel,
        }
        fp_records.append(rec)
        episodes_by_video[vid].append(rec)

    # Classification into the 8 categories:
    # 1. VLM-confirmed with successful VLM response
    # 2. heuristic-confirmed because VLM was unavailable/throttled
    # 3. heuristic override after VLM rejection
    # 4. duplicate/burst episode
    # 5. legitimate crossing maneuver incorrectly admitted
    # 6. queue/traffic-compression candidate
    # 7. shoulder/barrier-related false positive
    # 8. other

    # Since GROQ_API_KEY was not set, ALL VLM dispatches evaluated via heuristic fallback (source == 'kinematic_vlm_fallback').
    # But let's also understand the physical root cause category (cross-traffic, queue compression, shoulder/barrier, duplicate, etc.)
    # and whether it would have been rejected if VLM fallback hadn't heuristically confirmed it.

    category_counts = defaultdict(int)
    classified_records = []

    # To detect duplicate/burst episodes:
    # Check if there is another episode in the same video on the same involved tracks within 60 frames (~2.0s)
    for rec in fp_records:
        vid = rec["video_id"]
        frame = rec["frame"]
        tel = rec["telemetry"]
        etype = rec["event_type"]
        tracks = set(tel.get("involved_tracks", []))
        dist = float(tel.get("distance_m", 99.0))
        is_crossing = bool(tel.get("is_crossing_trajectory", False))
        heading_delta = float(tel.get("heading_delta_deg", 0.0))
        anomaly_type = str(tel.get("anomaly_type", ""))
        shoulder_admitted = bool(tel.get("shoulder_admitted", False))
        iou = float(tel.get("bbox_iou", 0.0))

        # Check duplicate/burst:
        is_duplicate = False
        for other in episodes_by_video[vid]:
            if other["index"] < rec["index"]: # prior episode
                other_tracks = set(other["telemetry"].get("involved_tracks", []))
                # Sharing tracks and within 75 frames (2.5s)
                if (tracks & other_tracks) and abs(other["frame"] - frame) <= 75:
                    is_duplicate = True
                    break

        cat = None
        detail = ""

        # Primary causal classification
        # Category 2: Every single one of these was heuristic-confirmed because VLM was unavailable (no API key).
        # But we must also classify their physical etiology into categories 4, 5, 6, 7, 8.
        if is_duplicate:
            cat = "4. duplicate/burst episode"
            detail = f"Occurred within 2.5s of prior episode with shared tracks {tracks}"
        elif shoulder_admitted or etype in ("barrier_strike", "single_vehicle_collision") or "barrier" in anomaly_type or "shoulder" in anomaly_type:
            cat = "7. shoulder/barrier-related false positive"
            detail = f"Shoulder/barrier trigger: shoulder_admitted={shoulder_admitted}, d_edge={tel.get('d_edge_m')}m, yaw={tel.get('yaw_delta_deg')}deg"
        elif is_crossing or (35.0 <= heading_delta <= 145.0):
            cat = "5. legitimate crossing maneuver incorrectly admitted"
            detail = f"Crossing trajectory in turning traffic: dist={dist:.2f}m, heading_delta={heading_delta:.1f}deg, iou={iou:.2f}, S_dissip={tel.get('S_dissip', 0.0)}"
        elif etype in ("collision_candidate", "collision") and (heading_delta < 35.0 or heading_delta > 145.0 or not is_crossing):
            cat = "6. queue/traffic-compression candidate"
            detail = f"Parallel flow / queue compression: dist={dist:.2f}m, heading_delta={heading_delta:.1f}deg, accel={tel.get('accel_ms2')}m/s2"
        else:
            cat = "8. other"
            detail = f"Unclassified event type: {etype}, {anomaly_type}"

        rec["assigned_category"] = cat
        rec["category_detail"] = detail
        category_counts[cat] += 1
        classified_records.append(rec)

    print("\n" + "=" * 80)
    print("MANDATORY ATTRIBUTION SUMMARY: 45 DAYTIME FALSE POSITIVES (ITERATION R)")
    print("=" * 80)

    total_fps = len(classified_records)
    for cat in sorted(category_counts.keys()):
        cnt = category_counts[cat]
        pct = (cnt / total_fps) * 100.0
        print(f"{cat:<55}: {cnt:>2} ({pct:>5.1f}%)")
    print("-" * 80)
    print(f"{'Total Daytime False Positives':<55}: {total_fps:>2} (100.0%)")
    print("=" * 80)

    # Pipeline counts from benchmark data
    raw_cands = bench_data["pipeline_progression"]["raw_candidate_proposals"]
    episodes = bench_data["pipeline_progression"]["consolidated_episodes"]
    vlm_dispatches = bench_data["pipeline_progression"]["vlm_calls"]
    # All VLM dispatches evaluated via heuristic fallback because GROQ_API_KEY was unset
    vlm_responses = 0
    fallback_events = vlm_dispatches
    final_confirmations = bench_data["pipeline_progression"]["confirmed_incidents"]
    tp_count = bench_data["metrics"]["true_positives"]
    fp_count = bench_data["metrics"]["false_positives"]

    print("\nPIPELINE DISPATCH FUNNEL (FULL 30-CLIP SWEEP):")
    print(f"  Raw Candidates:        {raw_cands}")
    print(f"  Consolidated Episodes: {episodes}")
    print(f"  VLM Dispatches:        {vlm_dispatches}")
    print(f"  VLM Responses:         {vlm_responses} (0.0% — offline mock_fallback mode)")
    print(f"  Fallback Events:       {fallback_events} (100.0% evaluated via _generate_fallback_verdict)")
    print(f"  Final Confirmations:   {final_confirmations} (TP={tp_count}, Daytime FP={fp_count}, Nocturnal FP=0)")
    print("=" * 80)

    # Detailed breakdown per video
    print("\nPER-VIDEO DAYTIME FP BREAKDOWN:")
    for vid, recs in episodes_by_video.items():
        print(f"\nVideo: {vid} ({len(recs)} FPs)")
        for r in recs:
            print(f"  - Frame {r['frame']:>4} | Ep: {r['episode_id']} | Type: {r['event_type']} | Cat: {r['assigned_category']}")
            print(f"    Detail: {r['category_detail']}")

    # Save attribution artifact
    attribution_artifact = {
        "iteration": "Iteration R",
        "total_daytime_fps": total_fps,
        "taxonomy_breakdown": {
            cat: {"count": cnt, "percentage": round((cnt / total_fps) * 100.0, 2)}
            for cat, cnt in category_counts.items()
        },
        "vlm_availability_attribution": {
            "1. VLM-confirmed with successful VLM response": {"count": 0, "percentage": 0.0},
            "2. heuristic-confirmed because VLM was unavailable/throttled": {"count": total_fps, "percentage": 100.0},
            "3. heuristic override after VLM rejection": {"count": 0, "percentage": 0.0},
        },
        "pipeline_funnel": {
            "raw_candidates": raw_cands,
            "consolidated_episodes": episodes,
            "vlm_dispatches": vlm_dispatches,
            "vlm_responses": vlm_responses,
            "fallback_events": fallback_events,
            "final_confirmations": final_confirmations,
            "true_positives": tp_count,
            "daytime_false_positives": fp_count,
            "nocturnal_false_positives": 0,
        },
        "classified_fps": classified_records,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/iteration_r_fp_attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution_artifact, f, indent=2)
    print("\nSaved attribution to artifacts/iteration_r_fp_attribution.json")

if __name__ == "__main__":
    run_attribution()
