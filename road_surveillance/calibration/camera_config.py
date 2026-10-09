"""Camera Configuration and Calibration Mode Definitions."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import numpy as np


class CameraCalibrationMode(str, Enum):
    """Explicit calibration mode for road surveillance cameras."""
    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"


@dataclass
class CameraConfiguration:
    """Explicit camera configuration specifying resolution, calibration status, and ground bounds."""
    camera_id: str
    mode: CameraCalibrationMode = CameraCalibrationMode.UNCALIBRATED
    calib_resolution: Optional[Tuple[int, int]] = None  # (width, height) at which source_points were defined
    source_points: Optional[List[List[float]]] = None
    destination_points: Optional[List[List[float]]] = None
    homography_matrix: Optional[List[List[float]]] = None
    lanes: List[Any] = field(default_factory=list)
    max_ground_range_m: float = 250.0
    min_ground_range_m: float = -50.0
    max_lateral_range_m: float = 150.0

    @property
    def is_calibrated(self) -> bool:
        """Return True if camera has surveyed ground-plane calibration."""
        return self.mode == CameraCalibrationMode.CALIBRATED

    @classmethod
    def create_uncalibrated(
        cls,
        camera_id: str,
        lanes: Optional[List[Any]] = None,
    ) -> "CameraConfiguration":
        """Create default uncalibrated configuration profile."""
        return cls(
            camera_id=camera_id,
            mode=CameraCalibrationMode.UNCALIBRATED,
            lanes=lanes or [],
        )

    @classmethod
    def create_calibrated(
        cls,
        camera_id: str,
        source_points: List[List[float]],
        destination_points: List[List[float]],
        calib_resolution: Tuple[int, int],
        lanes: Optional[List[Any]] = None,
    ) -> "CameraConfiguration":
        """Create explicit surveyed calibrated camera profile."""
        return cls(
            camera_id=camera_id,
            mode=CameraCalibrationMode.CALIBRATED,
            calib_resolution=calib_resolution,
            source_points=source_points,
            destination_points=destination_points,
            lanes=lanes or [],
        )

    @classmethod
    def from_config_dict(
        cls,
        camera_id: str,
        cfg_obj: Any,
        is_calibrated: bool = False,
    ) -> "CameraConfiguration":
        """Factory adapting AppConfig calibration section into structured CameraConfiguration."""
        if cfg_obj is None:
            return cls.create_uncalibrated(camera_id)

        source_pts = getattr(cfg_obj, "source_points", None)
        dest_pts = getattr(cfg_obj, "destination_points", None)
        h_mat = getattr(cfg_obj, "homography_matrix", None)
        lanes = getattr(cfg_obj, "lanes", [])
        calib_res = getattr(cfg_obj, "source_resolution", None)
        explicit_mode = getattr(cfg_obj, "is_calibrated", None)

        if explicit_mode is not None:
            mode = CameraCalibrationMode.CALIBRATED if explicit_mode else CameraCalibrationMode.UNCALIBRATED
        else:
            mode = CameraCalibrationMode.CALIBRATED if is_calibrated else CameraCalibrationMode.UNCALIBRATED

        # If source resolution was specified as a list, cast to tuple
        if calib_res and isinstance(calib_res, list) and len(calib_res) == 2:
            calib_res = (int(calib_res[0]), int(calib_res[1]))

        return cls(
            camera_id=camera_id,
            mode=mode,
            calib_resolution=calib_res,
            source_points=source_pts,
            destination_points=dest_pts,
            homography_matrix=h_mat,
            lanes=lanes,
        )
