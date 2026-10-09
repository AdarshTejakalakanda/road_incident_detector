"""Road Surface and Drivable Carriageway Detection Module.

Provides:
- Roadway Region of Interest (ROI) mask generation
- Perspective horizon / vanishing line estimation from vehicle geometry
- Trajectory-driven road occupancy heatmap accumulation
- Rejection of off-road detections (sky, tree canopies, overhead signage)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

logger = logging.getLogger("road_surveillance.calibration.road_surface")


@dataclass
class RoadBoundary:
    """Represents detected road spatial boundaries in image coordinates."""
    horizon_y: float                       # Estimated vertical horizon boundary (pixels)
    road_mask: Optional[np.ndarray] = None # Binary mask (H, W) where 255 = drivable roadway
    road_polygon: Optional[List[Tuple[float, float]]] = None  # [(x, y), ...] convex hull
    confidence: float = 0.0


class RoadSurfaceDetector:
    """Detects and maintains the drivable road surface boundary for surveillance views."""

    def __init__(
        self,
        default_horizon_ratio: float = 0.25,  # Upper 25% of frame assumed non-road by default
        min_road_samples: int = 10,
        enable_dynamic_horizon: bool = True,
    ):
        self.default_horizon_ratio = default_horizon_ratio
        self.min_road_samples = min_road_samples
        self.enable_dynamic_horizon = enable_dynamic_horizon

        # Running history of vehicle detections for road surface accumulation
        self.vehicle_ground_contacts: List[Tuple[float, float]] = []
        self.vehicle_boxes: List[Tuple[float, float, float, float]] = []  # (x, y, w, h)
        self.estimated_horizon_y: Optional[float] = None
        self._cached_mask: Optional[np.ndarray] = None
        self._last_frame_shape: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        """Reset internal accumulator (e.g. between videos)."""
        self.vehicle_ground_contacts.clear()
        self.vehicle_boxes.clear()
        self.estimated_horizon_y = None
        self._cached_mask = None
        self._last_frame_shape = None

    def register_vehicle_observations(
        self,
        boxes_xywh: List[Tuple[float, float, float, float]],
        frame_shape: Tuple[int, int],
    ) -> None:
        """Accumulate vehicle observations to refine roadway plane and horizon."""
        H, W = frame_shape[:2]
        self._last_frame_shape = (H, W)

        for bx, by, bw, bh in boxes_xywh:
            # Ground contact: bottom center of bounding box
            gc_x = bx + bw / 2.0
            gc_y = by + bh
            # Basic sanity filter: within image bounds
            if 0 <= gc_x < W and 0 <= gc_y < H:
                self.vehicle_ground_contacts.append((gc_x, gc_y))
                self.vehicle_boxes.append((bx, by, bw, bh))

        # Re-estimate horizon if sufficient observations
        if self.enable_dynamic_horizon and len(self.vehicle_boxes) >= self.min_road_samples:
            self._estimate_horizon_from_perspective(H)

    def _estimate_horizon_from_perspective(self, frame_height: int) -> None:
        """Estimate horizon y-coordinate via linear perspective scaling: height(y) ~ k * (y - y_horizon)."""
        if len(self.vehicle_boxes) < self.min_road_samples:
            return

        ys = []
        hs = []
        for bx, by, bw, bh in self.vehicle_boxes[-200:]:  # Use recent 200 samples
            # Bottom of box
            bottom_y = by + bh
            ys.append(bottom_y)
            hs.append(max(5.0, bh))

        ys = np.array(ys, dtype=np.float64)
        hs = np.array(hs, dtype=np.float64)

        # Robust linear fit: h = slope * y + intercept
        # Vanishing line is where h -> 0: y_horizon = -intercept / slope
        try:
            A = np.vstack([ys, np.ones(len(ys))]).T
            slope, intercept = np.linalg.lstsq(A, hs, rcond=None)[0]

            if slope > 0.01:
                y_horiz = -intercept / slope
                # Constrain horizon to plausible surveillance range: [0.05 * H, 0.55 * H]
                min_h = 0.05 * frame_height
                max_h = 0.55 * frame_height
                clamped_horiz = float(np.clip(y_horiz, min_h, max_h))
                self.estimated_horizon_y = clamped_horiz
            else:
                # Fallback to default ratio
                self.estimated_horizon_y = frame_height * self.default_horizon_ratio
        except Exception:
            self.estimated_horizon_y = frame_height * self.default_horizon_ratio

    def get_horizon_y(self, frame_height: int) -> float:
        """Get the current vertical horizon boundary coordinate in pixels."""
        if self.estimated_horizon_y is not None:
            return self.estimated_horizon_y
        return frame_height * self.default_horizon_ratio

    def is_point_on_road(self, px: float, py: float, frame_shape: Tuple[int, int]) -> bool:
        """Check if an image point (px, py) lies on the plausible roadway surface."""
        H, W = frame_shape[:2]

        # 1. Horizon rejection: anything above horizon is sky / background
        horiz_y = self.get_horizon_y(H)
        if py < horiz_y:
            return False

        # 2. Border margin check: plausible road within horizontal bounds
        if px < 0 or px >= W or py >= H:
            return False

        return True

    def is_box_on_road(self, bbox_xywh: Tuple[float, float, float, float], frame_shape: Tuple[int, int]) -> bool:
        """Check if a vehicle bounding box's ground contact is on the road."""
        bx, by, bw, bh = bbox_xywh
        gc_x = bx + bw / 2.0
        gc_y = by + bh
        return self.is_point_on_road(gc_x, gc_y, frame_shape)

    def filter_detections_on_road(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_labels: List[str],
        frame_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Filter out false detections located above the roadway horizon line."""
        if len(boxes_xyxy) == 0:
            return boxes_xyxy, confidences, class_ids, class_labels

        H, W = frame_shape[:2]
        horiz_y = self.get_horizon_y(H)

        keep_indices = []
        for i, box in enumerate(boxes_xyxy):
            bx1, by1, bx2, by2 = box
            gc_x = (bx1 + bx2) / 2.0
            gc_y = by2

            # Ground contact must be below the horizon line
            if gc_y >= horiz_y and 0 <= gc_x < W and gc_y <= H + 10:
                keep_indices.append(i)
            else:
                logger.debug(
                    f"Discarded off-road detection {class_labels[i]} (conf={confidences[i]:.2f}) at y={gc_y:.1f} < horizon {horiz_y:.1f}"
                )

        if len(keep_indices) == len(boxes_xyxy):
            return boxes_xyxy, confidences, class_ids, class_labels

        if not keep_indices:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        idx_arr = np.array(keep_indices, dtype=np.int32)
        return (
            boxes_xyxy[idx_arr],
            confidences[idx_arr],
            class_ids[idx_arr],
            [class_labels[i] for i in keep_indices],
        )
