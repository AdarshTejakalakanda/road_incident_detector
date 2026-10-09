"""Trajectory filtering and smoothing."""

from typing import List, Tuple
import numpy as np


class TrajectoryFilter:
    """Outlier-robust filter that preserves genuine sudden stops and deceleration."""

    def __init__(self, alpha: float = 0.6, max_jump_m: float = 8.0):
        self.alpha = alpha  # EMA weight for newest point
        self.max_jump_m = max_jump_m

    def filter_point(
        self,
        new_pt: Tuple[float, float],
        history: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Apply adaptive EMA smoothing while rejecting physically impossible single-frame jumps."""
        if not history:
            return new_pt

        prev_pt = history[-1]
        dist = np.hypot(new_pt[0] - prev_pt[0], new_pt[1] - prev_pt[1])

        # If jump is physically impossible for road vehicles in one frame (e.g. tracking glitch)
        if dist > self.max_jump_m:
            # Dampen jump towards previous position
            damped_x = prev_pt[0] + (new_pt[0] - prev_pt[0]) * 0.3
            damped_y = prev_pt[1] + (new_pt[1] - prev_pt[1]) * 0.3
            return (damped_x, damped_y)

        # Standard EMA
        smoothed_x = self.alpha * new_pt[0] + (1.0 - self.alpha) * prev_pt[0]
        smoothed_y = self.alpha * new_pt[1] + (1.0 - self.alpha) * prev_pt[1]
        return (smoothed_x, smoothed_y)
