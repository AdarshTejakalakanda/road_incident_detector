"""Dataset Benchmark Evaluation Runner."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np

from .ground_truth import GroundTruthLoader, VideoGroundTruth
from .metrics import VideoEvaluationResult, StratifiedMetricSummary, compute_stratified_summary, BenchmarkReporter
from ..events.base import EventCandidate

logger = logging.getLogger("road_surveillance.evaluation.runner")


class DatasetEvaluator:
    """Evaluates pipeline detection candidates against incident dataset ground truth."""

    def __init__(
        self,
        ground_truth_loader: GroundTruthLoader,
        temporal_tolerance_frames: int = 35,
    ):
        self.gt_loader = ground_truth_loader
        self.temporal_tolerance_frames = temporal_tolerance_frames
        self.results: List[VideoEvaluationResult] = []

    def evaluate_video(
        self,
        video_id: str,
        candidates: List[EventCandidate],
        fps: float,
    ) -> Optional[VideoEvaluationResult]:
        """Evaluate event candidates for a single video against ground truth."""
        gt = self.gt_loader.get_by_video_id(video_id)
        if not gt:
            logger.warning(f"No ground truth record found for video {video_id}; skipping evaluation.")
            return None

        # Filter collision candidates
        collision_candidates = [c for c in candidates if "collision" in c.event_type]

        is_tp = False
        is_fn = False
        fp_count = 0
        predicted_frame = None
        temporal_error = None
        latency_sec = None

        if collision_candidates:
            collision_candidates.sort(key=lambda c: c.frame_idx)

            matched_candidate = None
            for cand in collision_candidates:
                diff = cand.frame_idx - gt.accident_frame
                if abs(diff) <= self.temporal_tolerance_frames:
                    matched_candidate = cand
                    break

            if matched_candidate is not None:
                is_tp = True
                predicted_frame = matched_candidate.frame_idx
                temporal_error = predicted_frame - gt.accident_frame
                latency_sec = temporal_error / fps if fps > 0 else 0.0
                fp_count = len(collision_candidates) - 1
            else:
                is_fn = True
                fp_count = len(collision_candidates)
        else:
            is_fn = True
            fp_count = 0

        eval_result = VideoEvaluationResult(
            video_id=video_id,
            is_tp=is_tp,
            is_fn=is_fn,
            fp_count=fp_count,
            gt_frame=gt.accident_frame,
            predicted_frame=predicted_frame,
            temporal_error_frames=temporal_error,
            latency_sec=latency_sec,
            gt_type=gt.incident_type,
            weather=gt.weather,
            scene_layout=gt.scene_layout,
            day_time=gt.day_time,
            quality=gt.quality,
        )

        self.results.append(eval_result)
        return eval_result

    def generate_benchmark_report(self, dataset_split: str = "test") -> str:
        """Aggregate results and produce stratified Markdown report."""
        if not self.results:
            return "No evaluation results available."

        overall = compute_stratified_summary(self.results)

        stratified_groups: Dict[str, Dict[str, StratifiedMetricSummary]] = {
            "weather": {},
            "scene_layout": {},
            "day_time": {},
            "quality": {},
            "incident_type": {},
        }

        by_weather = defaultdict(list)
        by_layout = defaultdict(list)
        by_day_time = defaultdict(list)
        by_quality = defaultdict(list)
        by_type = defaultdict(list)

        for r in self.results:
            by_weather[r.weather].append(r)
            by_layout[r.scene_layout].append(r)
            by_day_time[r.day_time].append(r)
            by_quality[r.quality].append(r)
            by_type[r.gt_type].append(r)

        for k, v in by_weather.items():
            stratified_groups["weather"][k] = compute_stratified_summary(v)
        for k, v in by_layout.items():
            stratified_groups["scene_layout"][k] = compute_stratified_summary(v)
        for k, v in by_day_time.items():
            stratified_groups["day_time"][k] = compute_stratified_summary(v)
        for k, v in by_quality.items():
            stratified_groups["quality"][k] = compute_stratified_summary(v)
        for k, v in by_type.items():
            stratified_groups["incident_type"][k] = compute_stratified_summary(v)

        return BenchmarkReporter.format_markdown_report(
            overall=overall,
            stratified_groups=stratified_groups,
            results=self.results,
            dataset_split=dataset_split,
        )
