"""Shared Spatial Traffic Flow Field Infrastructure.

Learns dominant traffic-flow directions across spatial grid cells, persists flow models
across traffic lulls, and serves as shared geometric/flow infrastructure for both
Adaptive Wrong-Way detection and Road Hazard / Obstacle corroboration.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

logger = logging.getLogger("road_surveillance.events.traffic_flow")


@dataclass
class FlowDirectionModel:
    """Represents a learned and persisted traffic flow vector for a roadway zone."""
    direction_vector: Tuple[float, float]  # Normalized (dx, dy)
    coherence: float                       # Circular concentration [0, 1]
    sample_count: int                      # Number of vehicle observations
    is_persisted: bool = False             # Flag indicating mature, persisted model
    last_updated_frame: int = 0


class SpatialTrafficFlowField:
    """Maintains spatial grid of traffic flow vectors across the surveillance view."""

    def __init__(
        self,
        grid_rows: int = 4,
        grid_cols: int = 4,
        min_samples_to_persist: int = 5,
        min_coherence_to_persist: float = 0.75,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.min_samples_to_persist = min_samples_to_persist
        self.min_coherence_to_persist = min_coherence_to_persist

        # Grid cells: (r, c) -> list of normalized direction vectors [(dx, dy), ...]
        self._cell_vectors: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        # Tracking contributing vehicles per cell
        self._cell_track_ids: Dict[Tuple[int, int], Set[int]] = {}
        # Persisted flow models: (r, c) -> FlowDirectionModel
        self._persisted_models: Dict[Tuple[int, int], FlowDirectionModel] = {}
        # Global fallback model (aggregate across all cells)
        self._global_vectors: List[Tuple[float, float]] = []
        self._global_persisted_model: Optional[FlowDirectionModel] = None

    def reset(self) -> None:
        """Reset internal accumulator (preserves persisted models unless full clear)."""
        self._cell_vectors.clear()
        self._cell_track_ids.clear()
        self._global_vectors.clear()

    def full_reset(self) -> None:
        """Completely clear all models and history."""
        self._cell_vectors.clear()
        self._cell_track_ids.clear()
        self._persisted_models.clear()
        self._global_vectors.clear()
        self._global_persisted_model = None

    def _get_cell_coord(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        c = int(np.clip(norm_x * self.grid_cols, 0, self.grid_cols - 1))
        r = int(np.clip(norm_y * self.grid_rows, 0, self.grid_rows - 1))
        return (r, c)

    def observe_motion(
        self,
        norm_x: float,
        norm_y: float,
        heading_vec: Tuple[float, float],
        frame_idx: int,
        track_id: Optional[int] = None,
    ) -> None:
        """Register an observed moving vehicle's heading vector."""
        dx, dy = heading_vec
        norm = math.hypot(dx, dy)
        if norm < 1e-4:
            return

        unit_vec = (dx / norm, dy / norm)
        cell = self._get_cell_coord(norm_x, norm_y)

        if cell not in self._cell_vectors:
            self._cell_vectors[cell] = []
        self._cell_vectors[cell].append(unit_vec)
        if len(self._cell_vectors[cell]) > 100:
            self._cell_vectors[cell].pop(0)
        self._global_vectors.append(unit_vec)
        if len(self._global_vectors) > 200:
            self._global_vectors.pop(0)

        if track_id is not None:
            if cell not in self._cell_track_ids:
                self._cell_track_ids[cell] = set()
            self._cell_track_ids[cell].add(track_id)

        # Update cell model if sufficient observations
        self._update_cell_model(cell, frame_idx)
        self._update_global_model(frame_idx)

    def _compute_flow_stats(self, vectors: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], float, int]:
        """Compute circular mean direction and coherence from a list of unit vectors."""
        n = len(vectors)
        if n == 0:
            return ((0.0, 0.0), 0.0, 0)

        sum_x = sum(v[0] for v in vectors)
        sum_y = sum(v[1] for v in vectors)
        r = math.hypot(sum_x, sum_y)
        coherence = float(r / n)

        if r > 1e-4:
            mean_dir = (float(sum_x / r), float(sum_y / r))
        else:
            mean_dir = (0.0, 0.0)

        return (mean_dir, coherence, n)

    def _update_cell_model(self, cell: Tuple[int, int], frame_idx: int) -> None:
        vecs = self._cell_vectors[cell]
        contributing_tracks = self._cell_track_ids.get(cell, set())
        has_min_tracks = (len(contributing_tracks) == 0 or len(contributing_tracks) >= 3 or (len(contributing_tracks) >= 2 and len(vecs) >= 20))
        if len(vecs) >= self.min_samples_to_persist and has_min_tracks:
            mean_dir, coherence, count = self._compute_flow_stats(vecs)
            if coherence >= self.min_coherence_to_persist:
                self._persisted_models[cell] = FlowDirectionModel(
                    direction_vector=mean_dir,
                    coherence=coherence,
                    sample_count=count,
                    is_persisted=True,
                    last_updated_frame=frame_idx,
                )
            else:
                # If opposing or multi-directional flow enters the cell, coherence collapses.
                # Invalidate the model so mixed-traffic/bidirectional roadways do not falsely accuse traffic.
                self._persisted_models.pop(cell, None)

    def _update_global_model(self, frame_idx: int) -> None:
        if len(self._global_vectors) >= self.min_samples_to_persist:
            mean_dir, coherence, count = self._compute_flow_stats(self._global_vectors)
            if coherence >= (self.min_coherence_to_persist * 0.9):
                self._global_persisted_model = FlowDirectionModel(
                    direction_vector=mean_dir,
                    coherence=coherence,
                    sample_count=count,
                    is_persisted=True,
                    last_updated_frame=frame_idx,
                )

    def get_flow_model_for_position(self, norm_x: float, norm_y: float) -> Optional[FlowDirectionModel]:
        """Retrieve the persisted flow direction model for a normalized coordinate (x, y)."""
        cell = self._get_cell_coord(norm_x, norm_y)
        if cell in self._persisted_models and self._persisted_models[cell].is_persisted:
            return self._persisted_models[cell]
        return None

    def is_active_traffic_corridor(self, norm_x: float, norm_y: float) -> bool:
        """Check if the given normalized coordinate is within an active, learned traffic flow corridor."""
        model = self.get_flow_model_for_position(norm_x, norm_y)
        return model is not None and model.is_persisted and model.coherence >= self.min_coherence_to_persist

    def get_expected_heading(self, norm_x: float, norm_y: float) -> Optional[Tuple[float, float]]:
        """Return the expected unit heading vector for a coordinate, or None if unknown."""
        model = self.get_flow_model_for_position(norm_x, norm_y)
        if model is not None and model.is_persisted:
            return model.direction_vector
        return None

    def angular_deviation_from_flow(
        self,
        norm_x: float,
        norm_y: float,
        heading_vec: Tuple[float, float],
    ) -> Optional[float]:
        """Compute angular deviation (in degrees, [0, 180]) between heading_vec and learned flow."""
        exp = self.get_expected_heading(norm_x, norm_y)
        if exp is None:
            return None
        dot = exp[0] * heading_vec[0] + exp[1] * heading_vec[1]
        dot_clamped = max(-1.0, min(1.0, dot))
        angle_rad = math.acos(dot_clamped)
        return math.degrees(angle_rad)
