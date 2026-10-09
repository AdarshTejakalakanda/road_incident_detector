"""Evaluation Metrics and Benchmark Aggregators."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class VideoEvaluationResult:
    """Evaluation result for an individual incident video."""
    video_id: str
    is_tp: bool
    is_fn: bool
    fp_count: int
    gt_frame: int
    predicted_frame: Optional[int]
    temporal_error_frames: Optional[int]  # predicted_frame - gt_frame
    latency_sec: Optional[float]          # temporal_error_frames / fps
    gt_type: str
    weather: str
    scene_layout: str
    day_time: str
    quality: str


@dataclass
class StratifiedMetricSummary:
    """Metrics aggregated over a stratified subset of videos."""
    total_videos: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    fp_per_video: float = 0.0
    mean_temporal_error_frames: float = 0.0
    median_temporal_error_frames: float = 0.0
    mean_latency_sec: float = 0.0
    median_latency_sec: float = 0.0


def compute_stratified_summary(results: List[VideoEvaluationResult]) -> StratifiedMetricSummary:
    """Calculate aggregated TP/FP/FN, Precision, Recall, F1, and temporal latency metrics."""
    if not results:
        return StratifiedMetricSummary()

    tp = sum(1 for r in results if r.is_tp)
    fn = sum(1 for r in results if r.is_fn)
    fp = sum(r.fp_count for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_per_vid = fp / len(results) if len(results) > 0 else 0.0

    temp_errors = [r.temporal_error_frames for r in results if r.temporal_error_frames is not None]
    latencies = [r.latency_sec for r in results if r.latency_sec is not None]

    mean_te = float(np.mean(temp_errors)) if temp_errors else 0.0
    median_te = float(np.median(temp_errors)) if temp_errors else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    median_lat = float(np.median(latencies)) if latencies else 0.0

    return StratifiedMetricSummary(
        total_videos=len(results),
        tp=tp,
        fp=fp,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        f1_score=float(f1),
        fp_per_video=float(fp_per_vid),
        mean_temporal_error_frames=mean_te,
        median_temporal_error_frames=median_te,
        mean_latency_sec=mean_lat,
        median_latency_sec=median_lat,
    )


class BenchmarkReporter:
    """Formats benchmark evaluation results into Markdown and structured summaries."""

    @staticmethod
    def format_markdown_report(
        overall: StratifiedMetricSummary,
        stratified_groups: Dict[str, Dict[str, StratifiedMetricSummary]],
        results: List[VideoEvaluationResult],
        dataset_split: str,
    ) -> str:
        lines = [
            f"# Phase 1 Incident Detection Quantitative Benchmark Report",
            f"**Split Evaluated**: `{dataset_split}`  ",
            f"**Total Videos Analyzed**: {overall.total_videos}",
            "",
            "## 1. Overall Incident Performance Summary",
            "| Metric | Value | Interpretation |",
            "| :--- | :--- | :--- |",
            f"| **Evaluated Videos** | {overall.total_videos} | Total sample size |",
            f"| **True Positives (TP)** | {overall.tp} | Ground-truth accidents successfully triggered within tolerance |",
            f"| **False Positives (FP)** | {overall.fp} | Candidates triggered outside accident window |",
            f"| **False Negatives (FN)** | {overall.fn} | Ground-truth accidents missed by detector |",
            f"| **Precision** | **{overall.precision:.3f}** | TP / (TP + FP) |",
            f"| **Recall** | **{overall.recall:.3f}** | TP / (TP + FN) |",
            f"| **F1 Score** | **{overall.f1_score:.3f}** | Harmonic mean of Precision & Recall |",
            f"| **FP per Video** | **{overall.fp_per_video:.2f}** | Average false alarms per processed video |",
            f"| **Mean Temporal Error** | {overall.mean_temporal_error_frames:+.1f} frames | Average frame offset ($f_{{\\text{{pred}}}} - f_{{\\text{{gt}}}}$) |",
            f"| **Median Temporal Error** | {overall.median_temporal_error_frames:+.1f} frames | Robust central tendency |",
            f"| **Mean Detection Latency** | {overall.mean_latency_sec:+.2f} sec | Average seconds from impact to trigger |",
            f"| **Median Detection Latency** | {overall.median_latency_sec:+.2f} sec | Robust detection delay |",
            "",
            "---",
            "",
            "## 2. Stratified Robustness Evaluation",
        ]

        for group_name, group_dict in stratified_groups.items():
            lines.append(f"### By {group_name.replace('_', ' ').title()}")
            lines.append("| Category | Videos | TP | FP | FN | Precision | Recall | F1 | FP/Vid | Med Latency (s) |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for cat, s in sorted(group_dict.items()):
                lines.append(
                    f"| `{cat}` | {s.total_videos} | {s.tp} | {s.fp} | {s.fn} | "
                    f"{s.precision:.3f} | {s.recall:.3f} | {s.f1_score:.3f} | {s.fp_per_video:.2f} | {s.median_latency_sec:+.2f}s |"
                )
            lines.append("")

        lines.extend([
            "---",
            "",
            "## 3. Per-Video Evaluation Breakdown",
            "| Video ID | GT Frame | Pred Frame | Result | Temp Error | Latency | FP Count | Type | Weather | Quality | Layout |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for r in results:
            res_tag = "**HIT (TP)**" if r.is_tp else "_MISS (FN)_"
            pred_f = str(r.predicted_frame) if r.predicted_frame is not None else "-"
            t_err = f"{r.temporal_error_frames:+d} f" if r.temporal_error_frames is not None else "-"
            lat = f"{r.latency_sec:+.2f}s" if r.latency_sec is not None else "-"
            lines.append(
                f"| `{r.video_id}` | {r.gt_frame} | {pred_f} | {res_tag} | {t_err} | {lat} | {r.fp_count} | "
                f"{r.gt_type} | {r.weather} | {r.quality} | {r.scene_layout} |"
            )

        lines.append("")
        return "\n".join(lines)
