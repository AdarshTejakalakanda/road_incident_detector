"""Complete Collision Candidate Signal Ablation Runner with Kinematics Caching."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import ByteTrackTracker, Track
from road_surveillance.calibration import PlanarHomography, LaneManager
from road_surveillance.geometry import GeometryEngine
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from road_surveillance.evaluation import GroundTruthLoader
from road_surveillance.events import EventEngine

def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 1e-6 else 0.0


class AblationCollisionDetector:
    def __init__(self, mode: str):
        self.mode = mode
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.prev_distances: Dict[Tuple[int, int], float] = {}

    def reset(self):
        self.persistence_counts.clear()
        self.prev_distances.clear()

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
    ) -> List[Dict]:
        candidates = []
        current_pair_keys = set()
        VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

        for conv in convergences:
            t1_id = conv.track_id_1
            t2_id = conv.track_id_2
            k1 = kinematics.get(t1_id)
            k2 = kinematics.get(t2_id)
            tr1 = tracks.get(t1_id)
            tr2 = tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            dist = conv.distance_m
            prev_d = self.prev_distances.get(pair_key, dist)
            is_closing = (dist - prev_d) < -0.1
            self.prev_distances[pair_key] = dist

            iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            conv_signal = bool(conv.is_converging and conv.tca_sec < 2.0 and conv.dca_m < 3.5)
            decel_signal = bool(k1.acceleration_mag < -3.0 or k2.acceleration_mag < -3.0)

            strict_proximity = (dist <= 3.5) or (iou >= 0.25 and dist <= 10.0)
            loose_proximity = (dist <= 5.0) or (iou >= 0.10 and dist <= 15.0)

            triggered = False

            if self.mode == "1_distance_only":
                triggered = (dist <= 3.5)

            elif self.mode == "2_distance_iou":
                triggered = (dist <= 3.5 and iou >= 0.15) or (dist <= 1.5)

            elif self.mode == "3_distance_convergence":
                triggered = loose_proximity and (conv_signal or is_closing) and (dist <= 4.0 or iou >= 0.15)

            elif self.mode == "4_distance_convergence_accel":
                triggered = strict_proximity and (conv_signal or is_closing) and decel_signal

            elif self.mode == "5_distance_convergence_accel_persistence":
                cond = strict_proximity and (conv_signal or is_closing) and decel_signal
                if cond:
                    self.persistence_counts[pair_key] = self.persistence_counts.get(pair_key, 0) + 1
                else:
                    self.persistence_counts[pair_key] = 0
                triggered = (self.persistence_counts.get(pair_key, 0) >= 2)

            if triggered:
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "pair": pair_key,
                    "dist": dist,
                    "iou": iou,
                })

        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
                self.persistence_counts.pop(pk, None)

        return candidates


def cluster_episodes(raw_candidates: List[Dict], max_gap_frames: int = 25) -> List[Dict]:
    """Cluster raw triggers into distinct incident episodes."""
    if not raw_candidates:
        return []

    sorted_cands = sorted(raw_candidates, key=lambda c: c["frame_idx"])
    episodes = []
    current_ep = [sorted_cands[0]]

    for c in sorted_cands[1:]:
        if (c["frame_idx"] - current_ep[-1]["frame_idx"]) <= max_gap_frames:
            current_ep.append(c)
        else:
            episodes.append({
                "start_frame": current_ep[0]["frame_idx"],
                "end_frame": current_ep[-1]["frame_idx"],
                "trigger_frame": current_ep[0]["frame_idx"],
                "raw_count": len(current_ep),
            })
            current_ep = [c]

    if current_ep:
        episodes.append({
            "start_frame": current_ep[0]["frame_idx"],
            "end_frame": current_ep[-1]["frame_idx"],
            "trigger_frame": current_ep[0]["frame_idx"],
            "raw_count": len(current_ep),
        })

    return episodes


def evaluate_candidate_list(
    raw_candidates: List[Dict],
    gt_frame: int,
    fps: float,
    tolerance_frames: int = 35,
) -> Tuple[bool, int, Optional[int], Optional[float], int]:
    """Evaluates candidates for a single video against ground truth.
    Returns: (is_tp, fp_episodes_count, lead_time_frames, lead_time_sec, total_raw_fp)
    """
    episodes = cluster_episodes(raw_candidates)

    if not episodes:
        return False, 0, None, None, 0

    matched_episode = None
    fp_episodes = 0
    raw_fp = 0

    for ep in episodes:
        # Check if episode start_frame or any trigger falls within tolerance
        diff = ep["trigger_frame"] - gt_frame
        if abs(diff) <= tolerance_frames and matched_episode is None:
            matched_episode = ep
        else:
            fp_episodes += 1
            raw_fp += ep["raw_count"]

    if matched_episode is not None:
        is_tp = True
        lead_frames = gt_frame - matched_episode["trigger_frame"]  # Positive lead time = triggered before impact
        lead_sec = lead_frames / fps if fps > 0 else 0.0
        return is_tp, fp_episodes, lead_frames, lead_sec, raw_fp
    else:
        is_tp = False
        return is_tp, len(episodes), None, None, len(raw_candidates)


def run_ablation_study(
    config_path: str = "road_surveillance/config/default_config.yaml",
    split: str = "test",
    split_type: str = "split_in_distribution",
    max_videos: int = 15,
):
    cfg = load_config(config_path)
    gt_loader = GroundTruthLoader("C:/Users/KALAKA~1/Downloads/archive/metadata-real.csv")
    records = gt_loader.filter_split(split=split, split_type=split_type)

    detector = YOLO26Detector(cfg.detection.model_path, conf_threshold=cfg.detection.confidence_threshold, device=cfg.detection.device)
    tracker = ByteTrackTracker()

    ablation_modes = [
        ("0_baseline", "Baseline (Current Multi-Signal)"),
        ("1_distance_only", "Stage 1: Distance Only (<= 3.5m)"),
        ("2_distance_iou", "Stage 2: Distance + IoU (<= 3.5m & IoU >= 0.15)"),
        ("3_distance_convergence", "Stage 3: Distance + Convergence (Closing, TCA < 2s)"),
        ("4_distance_convergence_accel", "Stage 4: Distance + Convergence + Decel (a < -3m/s2)"),
        ("5_distance_convergence_accel_persistence", "Stage 5: Dist + Conv + Decel + Persistence (>= 2 frames)"),
    ]

    ablation_detectors = {
        mode_id: AblationCollisionDetector(mode_id)
        for mode_id, _ in ablation_modes if mode_id != "0_baseline"
    }

    baseline_engine = EventEngine(config=cfg.events, available_vocabulary=detector.names)

    # Cache for video frames kinematics
    video_data = []

    print(f"\n[1/2] Processing and caching {max_videos} videos from '{split_type}' ({split})...")
    for idx, gt_rec in enumerate(records[:max_videos], 1):
        v_name = os.path.basename(gt_rec.video_path)
        v_path = os.path.join(cfg.dataset.path, v_name)
        if not os.path.exists(v_path):
            continue

        print(f"  [{idx}/{max_videos}] Ingesting {gt_rec.video_id} (GT={gt_rec.accident_frame})...")
        ingestor = DatasetVideoIngestor(v_path)
        h = PlanarHomography()
        geom = GeometryEngine(h)
        tracker.reset()

        frame_records = []
        fps = 30.0

        for frame in ingestor.stream_video(v_path):
            fps = frame.fps
            batch = detector.detect(frame)
            tl, tracks = tracker.update(batch)
            kin, conv = geom.process_frame(tl, tracks)
            frame_records.append((frame.frame_idx, frame.pts_ns, dict(tracks), dict(kin), list(conv)))

        video_data.append({
            "video_id": gt_rec.video_id,
            "gt_frame": gt_rec.accident_frame,
            "fps": fps,
            "frames": frame_records,
        })

    print(f"\n[2/2] Running Ablation across {len(ablation_modes)} configurations on {len(video_data)} cached videos...")

    ablation_results = {}

    for mode_id, mode_label in ablation_modes:
        print(f"  Evaluating: {mode_label}...")
        tp_count = 0
        fn_count = 0
        fp_episodes_total = 0
        raw_fp_total = 0
        lead_times_frames = []
        lead_times_sec = []
        abs_temp_errors = []

        for v in video_data:
            gt_f = v["gt_frame"]
            fps = v["fps"]

            if mode_id == "0_baseline":
                baseline_engine.reset()
                cands = []
                for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                    c = baseline_engine.process_frame(f_idx, pts_ns, v["video_id"], tr, kin, conv, LaneManager())
                    colls = [x for x in c if "collision" in x.event_type]
                    for x in colls:
                        cands.append({"frame_idx": x.frame_idx, "pts_ns": x.timestamp_ns})
            else:
                det = ablation_detectors[mode_id]
                det.reset()
                cands = []
                for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                    c = det.evaluate_frame(f_idx, pts_ns, tr, kin, conv)
                    cands.extend(c)

            is_tp, fp_ep, lead_f, lead_s, r_fp = evaluate_candidate_list(cands, gt_f, fps, tolerance_frames=35)

            if is_tp:
                tp_count += 1
                lead_times_frames.append(lead_f)
                lead_times_sec.append(lead_s)
                abs_temp_errors.append(abs(lead_f))
            else:
                fn_count += 1

            fp_episodes_total += fp_ep
            raw_fp_total += r_fp

        total_vids = len(video_data)
        prec = tp_count / (tp_count + fp_episodes_total) if (tp_count + fp_episodes_total) > 0 else 0.0
        rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        raw_prec = tp_count / (tp_count + raw_fp_total) if (tp_count + raw_fp_total) > 0 else 0.0
        raw_f1 = (2 * raw_prec * rec) / (raw_prec + rec) if (raw_prec + rec) > 0 else 0.0

        med_lead_f = float(np.median(lead_times_frames)) if lead_times_frames else 0.0
        med_lead_s = float(np.median(lead_times_sec)) if lead_times_sec else 0.0
        med_abs_err = float(np.median(abs_temp_errors)) if abs_temp_errors else 0.0

        ablation_results[mode_id] = {
            "label": mode_label,
            "tp": tp_count,
            "fp_episodes": fp_episodes_total,
            "raw_fp": raw_fp_total,
            "fn": fn_count,
            "precision_ep": prec,
            "recall": rec,
            "f1_ep": f1,
            "fp_per_video_ep": fp_episodes_total / total_vids if total_vids > 0 else 0.0,
            "raw_precision": raw_prec,
            "raw_fp_per_video": raw_fp_total / total_vids if total_vids > 0 else 0.0,
            "med_lead_frames": med_lead_f,
            "med_lead_sec": med_lead_s,
            "med_abs_error": med_abs_err,
        }

    # Print Formatted Markdown Table
    report = [
        f"# Collision Candidate Signal Ablation Benchmark Results",
        f"**Split**: `{split_type}` (`{split}`) | **Videos Evaluated**: {len(video_data)}",
        "",
        "## 1. Episode-Deduplicated Performance Comparison",
        "| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Video | Med Lead Time | Med Abs Error |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for mode_id, _ in ablation_modes:
        r = ablation_results[mode_id]
        report.append(
            f"| **{r['label']}** | {r['tp']} | {r['fp_episodes']} | {r['fn']} | "
            f"**{r['precision_ep']:.3f}** | **{r['recall']:.3f}** | **{r['f1_ep']:.3f}** | "
            f"{r['fp_per_video_ep']:.2f} | +{r['med_lead_sec']:.2f}s ({r['med_lead_frames']:+.0f}f) | {r['med_abs_error']:.0f} frames |"
        )

    report.extend([
        "",
        "## 2. Impact of Event Deduplication: Episode-Level vs. Raw Trigger Noise",
        "| Configuration | Raw FP | Raw FP / Vid | Raw Precision | Ep FP | Ep FP / Vid | **Episode Precision** | Precision Gain |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for mode_id, _ in ablation_modes:
        r = ablation_results[mode_id]
        gain = (r['precision_ep'] / r['raw_precision']) if r['raw_precision'] > 0 else 1.0
        report.append(
            f"| **{r['label']}** | {r['raw_fp']} | {r['raw_fp_per_video']:.1f} | {r['raw_precision']:.3f} | "
            f"{r['fp_episodes']} | {r['fp_per_video_ep']:.2f} | **{r['precision_ep']:.3f}** | **{gain:.1f}x** |"
        )

    out_md = "\n".join(report)
    print("\n" + out_md + "\n")

    with open("ablation_benchmark_results.md", "w", encoding="utf-8") as f:
        f.write(out_md)
    print("Ablation results saved to: ablation_benchmark_results.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--split-type", default="split_in_distribution")
    parser.add_argument("--max-videos", type=int, default=15)
    args = parser.parse_args()

    run_ablation_study(split=args.split, split_type=args.split_type, max_videos=args.max_videos)
