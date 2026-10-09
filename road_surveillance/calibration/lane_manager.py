"""Lane Manager and Polygon Containment Module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class Lane:
    """Representation of a roadway lane or carriageway."""
    lane_id: str
    name: str
    is_active: bool
    permitted_direction: Tuple[float, float]  # Unit vector (dx, dy)
    polygon_ground: List[Tuple[float, float]]  # [(X, Y), ...] in meters
    polygon_pixel: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        # Normalize direction vector
        dx, dy = self.permitted_direction
        norm = np.hypot(dx, dy)
        if norm > 1e-6:
            self.permitted_direction = (dx / norm, dy / norm)
        else:
            self.permitted_direction = (0.0, 1.0)

        # Build shapely polygon for fast, robust Point-in-Polygon
        if len(self.polygon_ground) >= 3:
            self._shapely_poly = Polygon(self.polygon_ground)
        else:
            self._shapely_poly = None

    def contains_ground_point(self, X: float, Y: float) -> bool:
        """Check if ground coordinate (X, Y) lies inside the lane polygon."""
        if self._shapely_poly is None:
            return True
        pt = Point(X, Y)
        return self._shapely_poly.contains(pt) or self._shapely_poly.touches(pt)

    def contains_pixel_point(self, u: float, v: float) -> bool:
        """Check if pixel coordinate (u, v) lies inside the projected lane polygon."""
        if not self.polygon_pixel or len(self.polygon_pixel) < 3:
            return True
        poly_arr = np.array(self.polygon_pixel, dtype=np.int32)
        res = cv2.pointPolygonTest(poly_arr, (float(u), float(v)), False)
        return res >= 0


class LaneManager:
    """Manages active and inactive roadway lanes for a camera view."""

    def __init__(self, lanes: Optional[List[Lane]] = None, homography: Optional[Any] = None):
        self.lanes: List[Lane] = lanes or []
        self.homography = homography

    def add_lane(self, lane: Lane) -> None:
        self.lanes.append(lane)

    def get_lane_for_ground_point(self, X: float, Y: float) -> Optional[Lane]:
        """Find the first matching lane for a ground coordinate."""
        for lane in self.lanes:
            if lane.contains_ground_point(X, Y):
                return lane
        return None

    def is_in_active_lane(self, X: float, Y: float) -> bool:
        """Check if point is within any active carriageway/lane."""
        if not self.lanes:
            return True  # If no lanes configured, treat entire scene as active
        lane = self.get_lane_for_ground_point(X, Y)
        return lane is not None and lane.is_active

    @classmethod
    def from_config(cls, lane_configs, homography=None) -> "LaneManager":
        """Instantiate LaneManager from list of LaneConfig objects."""
        lanes = []
        for l_cfg in lane_configs:
            poly_g = [tuple(pt) for pt in l_cfg.polygon_ground]
            poly_px = None
            if homography is not None and len(poly_g) >= 3:
                pts_g_arr = np.array(poly_g, dtype=np.float64)
                poly_px_arr = homography.batch_ground_to_pixel(pts_g_arr)
                poly_px = [tuple(p) for p in poly_px_arr]

            lane = Lane(
                lane_id=l_cfg.lane_id,
                name=l_cfg.name,
                is_active=l_cfg.is_active,
                permitted_direction=tuple(l_cfg.permitted_direction),
                polygon_ground=poly_g,
                polygon_pixel=poly_px,
            )
            lanes.append(lane)
        return cls(lanes, homography=homography)
