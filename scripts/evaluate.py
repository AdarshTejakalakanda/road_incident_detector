import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import logging
import time
from typing import Optional

from road_surveillance.config import load_config, AppConfig
from road_surveillance.video import DatasetVideoIngestor
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import ByteTrackTracker
from road_surveillance.calibration import PlanarHomography, LaneManager
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine
from road_surveillance.evaluation import GroundTruthLoader, DatasetEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def evaluate_benchmark(
    config_path: str = "road_surveillance/config/default_config.yaml",
    metadata_path: Optional[str] = None,
    split: str = "test",
    split_type: str = "split_in_distribution",
    max_videos: Optional[int] = None,
    report_output_path: Optional[str] = "benchmark_report.md",
):
    """Execute quantitative collision benchmark evaluation against metadata-real.csv."""
    cfg: AppConfig = load_config(config_path)

    meta_file = metadata_path or cfg.evaluation.metadata_path
    if not os.path.exists(meta_file):
        logger.error(f"Ground truth metadata file does not exist: {meta_file}")
        return

    logger.info("=" * 65)
    logger.info("ROAD SURVEILLANCE INCIDENT DETECTION BENCHMARK EVALUATION")
    logger.info("=" * 65)
    logger.info(f"Metadata path: {meta_file}")
    logger.info(f"Split type: {split_type} | Split: {split}")
    logger.info(f"YOLO26 Model: {cfg.detection.model_path}")

    # 1. Load Ground Truth
    gt_loader = GroundTruthLoader(meta_file)
    split_records = gt_loader.filter_split(split=split, split_type=split_type)
    logger.info(f"Loaded {len(gt_loader.records)} total incident records.")
    logger.info(f"Selected {len(split_records)} records matching split '{split}' ({split_type}).")

    if not split_records:
        logger.error("No matching records found for specified split criteria.")
        return

    # 2. Initialize Evaluator
    evaluator = DatasetEvaluator(ground_truth_loader=gt_loader, temporal_tolerance_frames=35)

    # 3. Initialize YOLO26 Detector & Tracker
    detector = YOLO26Detector(
        model_path=cfg.detection.model_path,
        conf_threshold=cfg.detection.confidence_threshold,
        input_resolution=cfg.detection.input_resolution,
        device=cfg.detection.device,
    )

    tracker = ByteTrackTracker(
        track_activation_threshold=cfg.tracking.track_activation_threshold,
        lost_track_buffer=cfg.tracking.lost_track_buffer,
        minimum_matching_threshold=cfg.tracking.minimum_matching_threshold,
        frame_rate=cfg.tracking.frame_rate,
    )

    evaluated_count = 0
    start_eval_time = time.time()

    for idx, gt_rec in enumerate(split_records, 1):
        if max_videos and evaluated_count >= max_videos:
            break

        candidate_paths = [
            os.path.join(cfg.dataset.path, os.path.basename(gt_rec.video_path)),
            os.path.join(os.path.dirname(meta_file), gt_rec.video_path),
            gt_rec.video_path,
        ]

        resolved_video_path = None
        for cp in candidate_paths:
            if os.path.exists(cp):
                resolved_video_path = cp
                break

        if not resolved_video_path:
            logger.warning(f"Video file not found on disk for record {gt_rec.video_id}; skipping.")
            continue

        evaluated_count += 1
        video_id = gt_rec.video_id
        logger.info(
            f"Evaluating [{evaluated_count}/{len(split_records)}] {video_id} "
            f"(GT frame={gt_rec.accident_frame}, type={gt_rec.incident_type}, weather={gt_rec.weather})"
        )

        ingestor = DatasetVideoIngestor(resolved_video_path)
        calib_cfg = cfg.calibration.get(video_id) or cfg.calibration.get("default")
        homography = (
            PlanarHomography(
                homography_matrix=calib_cfg.homography_matrix,
                source_points=calib_cfg.source_points,
                destination_points=calib_cfg.destination_points,
            )
            if calib_cfg
            else PlanarHomography()
        )
        lane_manager = LaneManager.from_config(calib_cfg.lanes, homography) if calib_cfg else LaneManager()

        geometry_engine = GeometryEngine(homography=homography, lane_manager=lane_manager)
        event_engine = EventEngine(config=cfg.events, available_vocabulary=detector.names)
        tracker.reset()

        video_candidates = []
        video_fps = 30.0

        for frame in ingestor.stream_video(resolved_video_path):
            video_fps = frame.fps
            batch = detector.detect(frame)
            tracklets, active_tracks = tracker.update(batch)
            kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
            cands = event_engine.process_frame(
                frame_idx=frame.frame_idx,
                pts_ns=frame.pts_ns,
                video_id=video_id,
                tracks=active_tracks,
                kinematics=kinematics,
                convergences=convergences,
                lane_manager=lane_manager,
            )
            video_candidates.extend(cands)

        res = evaluator.evaluate_video(video_id=video_id, candidates=video_candidates, fps=video_fps)
        if res:
            status = "HIT (TP)" if res.is_tp else "MISS (FN)"
            t_err = f"{res.temporal_error_frames:+d} frames" if res.temporal_error_frames is not None else "N/A"
            logger.info(f"  -> Result: {status} | Temporal Error: {t_err} | FPs: {res.fp_count}")

    total_time = time.time() - start_eval_time
    logger.info(f"\nCompleted evaluation of {evaluated_count} videos in {total_time:.1f}s.")
    report_md = evaluator.generate_benchmark_report(dataset_split=f"{split} ({split_type})")

    print("\n" + report_md + "\n")

    if report_output_path:
        out_p = os.path.abspath(report_output_path)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Benchmark report saved to: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Incident Detection Benchmark against Ground Truth")
    parser.add_argument("--config", default="road_surveillance/config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--metadata", default=None, help="Path to metadata-real.csv")
    parser.add_argument("--split", default="test", choices=["test", "train"], help="Split subset")
    parser.add_argument("--split-type", default="split_in_distribution", choices=["split_in_distribution", "split_geo_aware"], help="Split strategy")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to evaluate")
    parser.add_argument("--report-out", default="benchmark_report.md", help="Output path for Markdown benchmark report")

    args = parser.parse_args()
    evaluate_benchmark(
        config_path=args.config,
        metadata_path=args.metadata,
        split=args.split,
        split_type=args.split_type,
        max_videos=args.max_videos,
        report_output_path=args.report_out,
    )
