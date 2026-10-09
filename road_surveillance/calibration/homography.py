"""Planar Homography and Dynamic Camera Calibration Module."""

import logging
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger("road_surveillance.calibration.homography")


class PlanarHomography:
    """Manages planar homography between image pixels (u, v) and metric ground plane (X, Y).
    
    Provides:
    - Resolution-aware dynamic adaptation (no hardcoded dimensions)
    - Horizon singularity and vanishing line detection
    - Physical coordinate boundary validation
    - Explicit calibrated vs uncalibrated operating modes
    """

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        source_points: Optional[List[List[float]]] = None,
        destination_points: Optional[List[List[float]]] = None,
        calib_resolution: Optional[Tuple[int, int]] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        is_calibrated: bool = False,
        camera_id: str = "default",
    ):
        self.calib_resolution = calib_resolution
        self.target_resolution = target_resolution or calib_resolution
        self.source_points = source_points
        self.destination_points = destination_points
        self.is_calibrated = is_calibrated
        self.camera_id = camera_id

        if homography_matrix is not None:
            self.H = np.array(homography_matrix, dtype=np.float64)
            if self.H.shape != (3, 3):
                raise ValueError(f"Homography matrix must be 3x3, got {self.H.shape}")
        elif source_points is not None and destination_points is not None:
            src = np.array(source_points, dtype=np.float32)
            dst = np.array(destination_points, dtype=np.float32)
            if len(src) < 4 or len(dst) < 4 or len(src) != len(dst):
                raise ValueError("Must provide at least 4 corresponding source and destination points")
            if len(src) == 4:
                self.H = cv2.getPerspectiveTransform(src, dst)
            else:
                self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            # Fallback identity transform
            self.H = np.eye(3, dtype=np.float64)

        # Canonicalize H so that points on the visible ground plane have positive projective depth (w > 0)
        if source_points and len(source_points) > 0:
            test_pt = np.array([source_points[0][0], source_points[0][1], 1.0], dtype=np.float64)
            if (self.H @ test_pt)[2] < 0:
                self.H = -self.H
        elif self.target_resolution:
            w_res, h_res = self.target_resolution
            test_pt = np.array([float(w_res) / 2.0, float(h_res), 1.0], dtype=np.float64)
            if (self.H @ test_pt)[2] < 0:
                self.H = -self.H

        # Invert H for ground -> pixel
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            logger.error("Homography matrix is singular and cannot be inverted!")
            self.H_inv = np.eye(3, dtype=np.float64)

    @property
    def horizon_line(self) -> Tuple[float, float, float]:
        """Return vanishing line coefficients (a, b, c) in image space where a*u + b*v + c = 0.
        
        In projective geometry, W'(u, v) = H[2, 0]*u + H[2, 1]*v + H[2, 2].
        Points satisfying W'(u, v) = 0 map to infinity.
        """
        return (float(self.H[2, 0]), float(self.H[2, 1]), float(self.H[2, 2]))

    def get_horizon_y(self, u: float) -> Optional[float]:
        """Compute the vertical pixel coordinate v on the horizon line for a given u."""
        a, b, c = self.horizon_line
        if abs(b) < 1e-8:
            return None
        return float(-(a * u + c) / b)

    def is_valid_ground_point(self, u: float, v: float) -> bool:
        """Check if image point (u, v) maps to a physically valid, plausible ground point.
        
        Rejects:
        1. Points at, above, or near the vanishing horizon line (W' <= 1e-4)
        2. Points that map to impossible ground plane coordinates (outside road bounds)
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        g_pt = self.H @ pt
        w = g_pt[2]
        if w <= 1e-4 or np.isnan(w) or np.isinf(w):
            return False
        X = g_pt[0] / w
        Y = g_pt[1] / w
        if np.isnan(X) or np.isnan(Y) or np.isinf(X) or np.isinf(Y):
            return False
        # Feasible surveillance road domain: lateral [-150m, 150m], longitudinal [-50m, 250m]
        if X < -150.0 or X > 150.0 or Y < -50.0 or Y > 250.0:
            return False
        return True

    def pixel_to_ground(self, u: float, v: float) -> Tuple[float, float]:
        """Convert single pixel coordinate (u, v) to ground coordinate (X, Y) in meters.
        
        Clamps values to physically plausible road domain and avoids horizon singularities.
        """
        pt = np.array([u, v, 1.0], dtype=np.float64)
        g_pt = self.H @ pt
        w = g_pt[2]
        if w <= 1e-4 or np.isnan(w) or np.isinf(w):
            return (0.0, 0.0)
        X = g_pt[0] / w
        Y = g_pt[1] / w
        X_clamped = float(np.clip(X, -150.0, 150.0))
        Y_clamped = float(np.clip(Y, -50.0, 250.0))
        return (X_clamped, Y_clamped)

    def pixel_to_ground_validated(self, u: float, v: float) -> Tuple[float, float, bool]:
        """Convert pixel (u, v) to ground coordinate (X, Y) with explicit validity status."""
        pt = np.array([u, v, 1.0], dtype=np.float64)
        g_pt = self.H @ pt
        w = g_pt[2]
        if w <= 1e-4 or np.isnan(w) or np.isinf(w):
            return (0.0, 0.0, False)
        X = g_pt[0] / w
        Y = g_pt[1] / w
        if np.isnan(X) or np.isnan(Y) or np.isinf(X) or np.isinf(Y):
            return (0.0, 0.0, False)
        is_valid = bool(-150.0 <= X <= 150.0 and -50.0 <= Y <= 250.0)
        X_clamped = float(np.clip(X, -150.0, 150.0))
        Y_clamped = float(np.clip(Y, -50.0, 250.0))
        return (X_clamped, Y_clamped, is_valid)

    def ground_to_pixel(self, X: float, Y: float) -> Tuple[float, float]:
        """Convert single ground coordinate (X, Y) in meters to image pixel (u, v)."""
        pt = np.array([X, Y, 1.0], dtype=np.float64)
        p_pt = self.H_inv @ pt
        if abs(p_pt[2]) < 1e-8:
            return (0.0, 0.0)
        u = p_pt[0] / p_pt[2]
        v = p_pt[1] / p_pt[2]
        return (float(u), float(v))

    def batch_pixel_to_ground(self, pts: np.ndarray) -> np.ndarray:
        """Batch convert pixel points (N, 2) to ground points (N, 2)."""
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])  # (N, 3)
        res = (self.H @ pts_homo.T).T  # (N, 3)
        w = res[:, 2:3]
        w_safe = np.where(np.abs(w) < 1e-4, 1e-4, w)
        res_xy = res[:, :2] / w_safe
        res_xy[:, 0] = np.clip(res_xy[:, 0], -150.0, 150.0)
        res_xy[:, 1] = np.clip(res_xy[:, 1], -50.0, 250.0)
        return res_xy

    def batch_ground_to_pixel(self, pts: np.ndarray) -> np.ndarray:
        """Batch convert ground points (N, 2) to pixel points (N, 2)."""
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float64)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        res = (self.H_inv @ pts_homo.T).T
        w = res[:, 2:3]
        w[np.abs(w) < 1e-8] = 1e-8
        return res[:, :2] / w

    def adapt_to_resolution(self, target_resolution: Tuple[int, int]) -> "PlanarHomography":
        """Create a new PlanarHomography adapted to target video resolution without hardcoding."""
        w_tgt, h_tgt = target_resolution
        if w_tgt <= 0 or h_tgt <= 0:
            raise ValueError(f"Invalid target resolution: {target_resolution}")

        if self.target_resolution == target_resolution and self.H is not None:
            return self

        # Determine reference resolution
        ref_res = self.calib_resolution or self.target_resolution
        if ref_res is None and self.source_points:
            max_x = max(pt[0] for pt in self.source_points)
            max_y = max(pt[1] for pt in self.source_points)
            if max_x <= 1.01 and max_y <= 1.01:
                ref_res = (1, 1)
            else:
                ref_res = (int(max(1.0, max_x)), int(max(1.0, max_y)))

        if ref_res is not None and self.source_points and self.destination_points:
            w_ref, h_ref = ref_res
            scale_x = float(w_tgt) / float(w_ref)
            scale_y = float(h_tgt) / float(h_ref)
            scaled_src = [
                [float(pt[0]) * scale_x, float(pt[1]) * scale_y]
                for pt in self.source_points
            ]
            return PlanarHomography(
                source_points=scaled_src,
                destination_points=self.destination_points,
                calib_resolution=self.calib_resolution,
                target_resolution=target_resolution,
                is_calibrated=self.is_calibrated,
                camera_id=self.camera_id,
            )

        # If only homography matrix exists without source points, rescale matrix directly
        if ref_res is not None:
            w_ref, h_ref = ref_res
            sx = float(w_ref) / float(w_tgt)
            sy = float(h_ref) / float(h_tgt)
            S = np.array([
                [sx, 0.0, 0.0],
                [0.0, sy, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            new_H = self.H @ S
            return PlanarHomography(
                homography_matrix=new_H,
                calib_resolution=self.calib_resolution,
                target_resolution=target_resolution,
                is_calibrated=self.is_calibrated,
                camera_id=self.camera_id,
            )

        return PlanarHomography(
            homography_matrix=self.H.copy(),
            calib_resolution=self.calib_resolution,
            target_resolution=target_resolution,
            is_calibrated=self.is_calibrated,
            camera_id=self.camera_id,
        )

    @classmethod
    def create_dynamic_homography(
        cls,
        source_points: List[List[float]],
        destination_points: List[List[float]],
        calib_resolution: Optional[Tuple[int, int]] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        is_calibrated: bool = False,
        camera_id: str = "default",
    ) -> "PlanarHomography":
        """Factory creating homography adapted to target resolution dynamically."""
        w_calib, h_calib = calib_resolution if calib_resolution else (1280, 720)
        w_target, h_target = target_resolution if target_resolution else (w_calib, h_calib)

        if w_calib <= 0 or h_calib <= 0 or (w_calib == w_target and h_calib == h_target):
            scaled_src = source_points
        else:
            scale_x = float(w_target) / float(w_calib)
            scale_y = float(h_target) / float(h_calib)
            scaled_src = [
                [float(pt[0]) * scale_x, float(pt[1]) * scale_y]
                for pt in source_points
            ]

        return cls(
            source_points=scaled_src,
            destination_points=destination_points,
            calib_resolution=(w_calib, h_calib),
            target_resolution=(w_target, h_target),
            is_calibrated=is_calibrated,
            camera_id=camera_id,
        )

    @classmethod
    def from_camera_config(
        cls,
        camera_cfg: Any,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> "PlanarHomography":
        """Factory building dynamic homography from CameraConfiguration."""
        if camera_cfg is None:
            return cls(is_calibrated=False)

        source_pts = getattr(camera_cfg, "source_points", None)
        dest_pts = getattr(camera_cfg, "destination_points", None)
        h_mat = getattr(camera_cfg, "homography_matrix", None)
        calib_res = getattr(camera_cfg, "calib_resolution", None)
        is_calib = getattr(camera_cfg, "is_calibrated", False)
        cam_id = getattr(camera_cfg, "camera_id", "default")

        if source_pts and dest_pts:
            base_h = cls(
                source_points=source_pts,
                destination_points=dest_pts,
                calib_resolution=calib_res,
                is_calibrated=is_calib,
                camera_id=cam_id,
            )
        elif h_mat is not None:
            base_h = cls(
                homography_matrix=np.array(h_mat, dtype=np.float64),
                calib_resolution=calib_res,
                is_calibrated=is_calib,
                camera_id=cam_id,
            )
        else:
            base_h = cls(is_calibrated=False, camera_id=cam_id)

        if target_resolution is not None:
            return base_h.adapt_to_resolution(target_resolution)
        return base_h

    def validate_calibration(
        self,
        ref_src: Optional[List[List[float]]] = None,
        ref_dst: Optional[List[List[float]]] = None,
        image_resolution: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Validate homography matrix numerical properties, horizon positioning, and reprojection accuracy."""
        det = float(np.linalg.det(self.H))
        cond_num = float(np.linalg.cond(self.H))
        is_invertible = bool(abs(det) > 1e-12 and cond_num < 1e8)

        # Horizon line check
        a, b, c = self.horizon_line
        horizon_info: Dict[str, Any] = {
            "coefficients": (a, b, c),
            "is_horizontal_enough": bool(abs(b) > 1e-8),
        }

        # Check if horizon position is geometrically valid in surveillance view
        horizon_valid = True
        img_res = image_resolution or self.target_resolution
        if img_res is not None and abs(b) > 1e-8:
            w_img, h_img = img_res
            v_horizon_center = -(a * (w_img / 2.0) + c) / b
            horizon_info["v_horizon_center"] = float(v_horizon_center)
            # Horizon should not cut deeply through the bottom driving lane (> 85% of image height)
            if v_horizon_center > 0.85 * h_img:
                horizon_valid = False

        results = {
            "determinant": det,
            "condition_number": cond_num,
            "is_invertible": is_invertible,
            "is_calibrated": self.is_calibrated,
            "horizon": horizon_info,
            "is_valid": bool(is_invertible and horizon_valid),
            "reprojection_error_rms": 0.0,
        }

        eval_src = ref_src or self.source_points
        eval_dst = ref_dst or self.destination_points
        if eval_src is not None and eval_dst is not None and len(eval_src) > 0:
            src_arr = np.array(eval_src, dtype=np.float64)
            dst_arr = np.array(eval_dst, dtype=np.float64)
            predicted_dst = self.batch_pixel_to_ground(src_arr)
            err = np.linalg.norm(predicted_dst - dst_arr, axis=1)
            results["reprojection_error_rms"] = float(np.sqrt(np.mean(err**2)))

        return results

    def draw_projected_grid(
        self,
        image: np.ndarray,
        x_range: Tuple[float, float] = (-15.0, 15.0),
        y_range: Tuple[float, float] = (0.0, 60.0),
        step_m: float = 10.0,
        color: Tuple[int, int, int] = (0, 255, 255),
    ) -> np.ndarray:
        """Project a regular metric ground grid onto the perspective image for calibration verification."""
        canvas = image.copy()
        xs = np.arange(x_range[0], x_range[1] + step_m, step_m)
        ys = np.arange(y_range[0], y_range[1] + step_m, step_m)

        # Draw lines of constant X
        for x in xs:
            pts_g = np.array([[x, y] for y in np.linspace(y_range[0], y_range[1], 20)])
            pts_px = self.batch_ground_to_pixel(pts_g).astype(np.int32)
            cv2.polylines(canvas, [pts_px], isClosed=False, color=color, thickness=1)

        # Draw lines of constant Y
        for y in ys:
            pts_g = np.array([[x, y] for x in np.linspace(x_range[0], x_range[1], 20)])
            pts_px = self.batch_ground_to_pixel(pts_g).astype(np.int32)
            cv2.polylines(canvas, [pts_px], isClosed=False, color=color, thickness=1)

        return canvas
