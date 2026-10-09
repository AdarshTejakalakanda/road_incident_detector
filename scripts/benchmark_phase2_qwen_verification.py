"""Phase 2 Controlled Experiment 1: Multimodal Verification Benchmark with Qwen3.8-27B.

Controlled Protocol:
- Same 30 frozen Phase 1 benchmark videos (15 in-distribution + 15 geo-aware).
- Same frozen candidate episodes (12 TP episodes + 22 FP episodes = 34 candidate episodes).
- Downstream Qwen3.8-27B multimodal verification on multi-frame visual keyframes + telemetry.
- Measures:
    1. How many of the 12 TP incidents remain CONFIRMED.
    2. How many of the 22 FP episodes are REJECTED.
    3. How many are marked UNCERTAIN.
    4. Post-verification Precision, Recall, F1, and FP episodes / video.
"""

import os
import sys
import json
import pickle
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config
from road_surveillance.verification.schema import VerificationVerdict, IncidentVerificationResult
from road_surveillance.verification.evidence_extractor import CandidateEvidenceExtractor
from road_surveillance.verification.qwen_verifier import QwenMultimodalVerifier
from scripts.benchmark_iteration_b_detector_recovery import cluster_episodes


def extract_candidate_telemetry(
    v_frames: List[Tuple],
    cand_frame: int,
    track_ids: List[int],
    detector: str,
) -> Dict[str, Any]:
    """Extract kinematic and geometric signals for the candidate episode."""
    tel: Dict[str, Any] = {
        "detector": detector,
        "track_ids": track_ids,
        "cand_frame": cand_frame,
    }

    # Find the frame in cached frames
    frame_data = None
    for f_idx, pts_ns, tracks, kin, conv in v_frames:
        if f_idx == cand_frame:
            frame_data = (f_idx, pts_ns, tracks, kin, conv)
            break

    if not frame_data:
        return tel

    f_idx, pts_ns, tracks, kin, conv = frame_data

    # Extract speeds and accelerations for involved tracks
    if len(track_ids) == 1:
        tid = track_ids[0]
        if tid in kin:
            k = kin[tid]
            tel["metric_speed_kmh"] = float(k.speed_kmh)
            tel["metric_accel_ms2"] = float(k.acceleration_mag)
            tel["yaw_rate_deg_s"] = float(getattr(k, "yaw_rate_deg_s", 0.0))
            tel["aspect_ratio_change"] = float(getattr(k, "aspect_ratio_change", 0.0))
    elif len(track_ids) >= 2:
        t1, t2 = track_ids[0], track_ids[1]
        if t1 in kin:
            tel["speed_1_kmh"] = float(kin[t1].speed_kmh)
            tel["accel_1_ms2"] = float(kin[t1].acceleration_mag)
        if t2 in kin:
            tel["speed_2_kmh"] = float(kin[t2].speed_kmh)
            tel["accel_2_ms2"] = float(kin[t2].acceleration_mag)

        # Check convergences for TCA and DCA
        if conv:
            for c in conv:
                if (c.track_id_1 == t1 and c.track_id_2 == t2) or (c.track_id_1 == t2 and c.track_id_2 == t1):
                    tel["tca_sec"] = float(c.tca_sec)
                    tel["dca_norm"] = float(c.dca_m)
                    break

    return tel


def run_controlled_experiment(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "Qwen/Qwen3.8-27B",
    save_evidence_dir: str = "./evidence/phase2_candidates",
    mock_mode: bool = False,
):
    print("=" * 70)
    print("PHASE 2 CONTROLLED EXPERIMENT: QWEN3.8-27B MULTIMODAL VERIFICATION")
    print("=" * 70)

    extractor = CandidateEvidenceExtractor()
    os.makedirs(save_evidence_dir, exist_ok=True)

    verifier = QwenMultimodalVerifier(
        api_base=api_base,
        api_key=api_key,
        model=model,
        mock_mode=mock_mode,
    )

    # Load frozen Phase 1 validation metadata
    with open("telemetry_phase1_final_validation.json", "r", encoding="utf-8") as f:
        phase1_meta = json.load(f)

    splits = ["split_in_distribution", "split_geo_aware"]
    all_episodes_results = []

    # Counters
    phase1_tp_count = 0
    phase1_fp_count = 0

    tp_confirmed = 0
    tp_rejected = 0
    tp_uncertain = 0

    fp_confirmed = 0
    fp_rejected = 0
    fp_uncertain = 0

    for split_type in splits:
        cache_file = f"cache_videos_iter_b_{split_type}_15.pkl"
        if not os.path.exists(cache_file):
            print(f"Warning: Cache file not found: {cache_file}")
            continue

        with open(cache_file, "rb") as f:
            videos = pickle.load(f)

        print(f"\nProcessing {split_type} ({len(videos)} videos)...")

        for v in videos:
            vid_id = v["video_id"]
            gt_f = v["gt_frame"]
            fps = v["fps"]

            # Find matching video in phase 1 telemetry
            meta_match = next((x for x in phase1_meta["all_videos"] if x["video_id"] == vid_id and x["split"] == ("in_distribution" if "in_distribution" in split_type else "geo_aware")), None)
            if not meta_match:
                continue

            raw_cands = [
                {"frame_idx": c["f"], "track_ids": c["tids"], "detector": c["det"]}
                for c in meta_match.get("candidates", [])
            ]
            episodes = cluster_episodes(raw_cands, max_gap_frames=25)

            if not episodes:
                continue

            # In Phase 1 temporal evaluation: first episode matching GT +/-35f is TP, all others are FP
            tp_found = False
            for ep in episodes:
                cand_f = ep["trigger_frame"]
                diff_f = cand_f - gt_f
                is_phase1_tp = (abs(diff_f) <= 35) and (not tp_found)

                if is_phase1_tp:
                    tp_found = True
                    phase1_status = "TP"
                    phase1_tp_count += 1
                else:
                    phase1_status = "FP"
                    phase1_fp_count += 1

                detector_name = ep["representative"]["detector"]
                track_ids = ep["representative"]["track_ids"]

                # Extract bounding boxes at candidate frame
                track_boxes = {}
                for f_idx, pts_ns, tracks, kin, conv in v["frames"]:
                    if f_idx == cand_f:
                        for tid, tr in tracks.items():
                            track_boxes[tid] = tr.current_bbox_xywh
                        break

                # Extract kinematic telemetry
                telemetry = extract_candidate_telemetry(v["frames"], cand_f, track_ids, detector_name)

                # Extract multi-frame evidence package
                evidence_pkg = extractor.extract_evidence_package(
                    video_id=vid_id,
                    candidate_frame=cand_f,
                    candidate_detector=detector_name,
                    involved_track_ids=track_ids,
                    telemetry=telemetry,
                    track_boxes=track_boxes,
                )

                # Save keyframe triplet for auditability
                ep_evidence_dir = os.path.join(save_evidence_dir, f"{vid_id}_f{cand_f}_{phase1_status}")
                os.makedirs(ep_evidence_dir, exist_ok=True)
                for k_name, k_img in evidence_pkg.keyframes.items():
                    if k_img is not None and k_img.size > 0:
                        cv2.imwrite(os.path.join(ep_evidence_dir, f"frame_{k_name}_{evidence_pkg.keyframe_indices.get(k_name, 0)}.jpg"), k_img)

                # Run downstream Qwen verification
                t0 = time.time()
                verif_res = verifier.verify(evidence_pkg)
                dt = time.time() - t0

                # Tabulate stats
                if phase1_status == "TP":
                    if verif_res.is_confirmed:
                        tp_confirmed += 1
                    elif verif_res.is_rejected:
                        tp_rejected += 1
                    else:
                        tp_uncertain += 1
                else:  # Phase 1 FP
                    if verif_res.is_rejected:
                        fp_rejected += 1
                    elif verif_res.is_confirmed:
                        fp_confirmed += 1
                    else:
                        fp_uncertain += 1

                record = {
                    "video_id": vid_id,
                    "split": split_type,
                    "gt_frame": gt_f,
                    "candidate_frame": cand_f,
                    "diff_frames": diff_f,
                    "phase1_status": phase1_status,
                    "detector": detector_name,
                    "track_ids": track_ids,
                    "verdict": verif_res.verdict.value,
                    "confidence": verif_res.confidence,
                    "crash_type": verif_res.detected_crash_type,
                    "rationale": verif_res.rationale,
                    "latency_sec": round(dt, 2),
                }
                all_episodes_results.append(record)

                print(f"  [{phase1_status}] {vid_id} @ f={cand_f} (GT={gt_f}, diff={diff_f:+d}f, det={detector_name}): "
                      f"Verdict={verif_res.verdict.value} (conf={verif_res.confidence:.2f}, crash={verif_res.detected_crash_type})")

    # Metrics computation
    print("\n" + "=" * 70)
    print("PHASE 2 MULTIMODAL VERIFICATION CONTROLLED RESULTS")
    print("=" * 70)

    print(f"\n1. Phase 1 Frozen Candidate Input:")
    print(f"   Total Candidate Episodes: {phase1_tp_count + phase1_fp_count}")
    print(f"   - Phase 1 TP Episodes:    {phase1_tp_count} (across 30 videos)")
    print(f"   - Phase 1 FP Episodes:    {phase1_fp_count} (across 30 videos)")

    print(f"\n2. Qwen3.8-27B Verification Breakdown:")
    print(f"   TP Episodes (N = {phase1_tp_count}):")
    print(f"     * CONFIRMED (Recall Kept):     {tp_confirmed} ({tp_confirmed / max(1, phase1_tp_count) * 100:.1f}%)")
    print(f"     * REJECTED (Recall Lost):      {tp_rejected} ({tp_rejected / max(1, phase1_tp_count) * 100:.1f}%)")
    print(f"     * UNCERTAIN:                   {tp_uncertain} ({tp_uncertain / max(1, phase1_tp_count) * 100:.1f}%)")
    print(f"   FP Episodes (N = {phase1_fp_count}):")
    print(f"     * REJECTED (FP Suppressed):    {fp_rejected} ({fp_rejected / max(1, phase1_fp_count) * 100:.1f}%)")
    print(f"     * CONFIRMED (FP Retained):     {fp_confirmed} ({fp_confirmed / max(1, phase1_fp_count) * 100:.1f}%)")
    print(f"     * UNCERTAIN:                   {fp_uncertain} ({fp_uncertain / max(1, phase1_fp_count) * 100:.1f}%)")

    # Evaluate under two operational policies for UNCERTAIN:
    # Policy A: Conservative ITS operator policy (treat UNCERTAIN as alert / keep)
    tp_a = tp_confirmed + tp_uncertain
    fp_a = fp_confirmed + fp_uncertain
    fn_a = 18 + (phase1_tp_count - tp_a)
    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
    rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
    f1_a = (2 * prec_a * rec_a) / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0.0

    # Policy B: Selective automated dispatch policy (only trigger on CONFIRMED)
    tp_b = tp_confirmed
    fp_b = fp_confirmed
    fn_b = 18 + (phase1_tp_count - tp_b)
    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
    rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
    f1_b = (2 * prec_b * rec_b) / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0.0

    print("\n3. Comparative Evaluation (Phase 1 vs. Phase 2 Post-Verification):")
    print("-" * 75)
    print(f"{'Metric':<30} | {'Phase 1 (Frozen)':<16} | {'Phase 2 (Policy A)':<18} | {'Phase 2 (Policy B)':<18}")
    print("-" * 75)
    print(f"{'True Positives (TP)':<30} | {phase1_tp_count:<16} | {tp_a:<18} | {tp_b:<18}")
    print(f"{'False Positive Episodes':<30} | {phase1_fp_count:<16} | {fp_a:<18} | {fp_b:<18}")
    print(f"{'False Negatives (FN)':<30} | {18:<16} | {fn_a:<18} | {fn_b:<18}")
    print(f"{'Precision':<30} | {phase1_tp_count / (phase1_tp_count + phase1_fp_count):<16.3f} | {prec_a:<18.3f} | {prec_b:<18.3f}")
    print(f"{'Recall':<30} | {phase1_tp_count / 30 * 100:<15.1f}% | {rec_a * 100:<17.1f}% | {rec_b * 100:<17.1f}%")
    print(f"{'F1-Score':<30} | {0.375:<16.3f} | {f1_a:<18.3f} | {f1_b:<18.3f}")
    print(f"{'FP Episodes / Video':<30} | {phase1_fp_count / 30:<16.2f} | {fp_a / 30:<18.2f} | {fp_b / 30:<18.2f}")
    print("-" * 75)

    # Save telemetry JSON
    output_telemetry = {
        "benchmark_summary": {
            "total_videos": 30,
            "phase1_tp": phase1_tp_count,
            "phase1_fp": phase1_fp_count,
            "tp_confirmed": tp_confirmed,
            "tp_rejected": tp_rejected,
            "tp_uncertain": tp_uncertain,
            "fp_rejected": fp_rejected,
            "fp_confirmed": fp_confirmed,
            "fp_uncertain": fp_uncertain,
            "policy_a": {
                "precision": prec_a,
                "recall": rec_a,
                "f1": f1_a,
                "fp_per_video": fp_a / 30,
            },
            "policy_b": {
                "precision": prec_b,
                "recall": rec_b,
                "f1": f1_b,
                "fp_per_video": fp_b / 30,
            },
        },
        "all_episodes": all_episodes_results,
    }

    out_json = "telemetry_phase2_qwen_verification.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_telemetry, f, indent=2)
    print(f"\nSaved verification telemetry to: {out_json}")
    print(f"Candidate keyframe packages saved to: {save_evidence_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Controlled Verification Benchmark")
    parser.add_argument("--api-base", type=str, default=None, help="OpenAI-compatible API base URL for Qwen3.8-27B")
    parser.add_argument("--api-key", type=str, default=None, help="API key for Qwen3.8-27B")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.8-27B", help="Model identifier")
    parser.add_argument("--mock", action="store_true", help="Run with mock verifier for pipeline validation")
    parser.add_argument("--save-evidence-dir", type=str, default="./evidence/phase2_candidates", help="Directory to save candidate keyframe packages")
    args = parser.parse_args()

    run_controlled_experiment(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        save_evidence_dir=args.save_evidence_dir,
        mock_mode=args.mock,
    )
