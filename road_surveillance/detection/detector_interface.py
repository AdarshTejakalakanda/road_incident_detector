"""Detector Interface and Data Structures."""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class DetectionBatch:
    """Detection results for a single frame."""
    video_id: str
    frame_idx: int
    pts_ns: int
    boxes: np.ndarray        # Shape [N, 4] in [x1, y1, x2, y2] pixel coordinates
    confidences: np.ndarray  # Shape [N] in range [0.0, 1.0]
    class_ids: np.ndarray    # Shape [N] integer class IDs
    class_labels: List[str]  # Human-readable class names
