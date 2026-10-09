"""Illumination estimation and low-light preprocessing ablation module.

Implements the N0-N3 low-light ablation ladder:
- N0: Raw untouched baseline
- N1: Contrast-Limited Adaptive Histogram Equalization (CLAHE) on L/Y channel
- N2: Gamma Power-Law Luminance Compression (gamma = 0.65) via fast LUT
- N3: Adaptive Selection (enhances only when mean luminance < threshold)
"""

import time
from typing import Dict, Any, Tuple, Literal, Optional
import numpy as np
import cv2


class IlluminationEstimator:
    """Estimates scene illumination level from video frames."""

    @staticmethod
    def estimate_mean_luminance(frame_bgr: np.ndarray) -> float:
        """Calculate mean scene luminance across the Y channel in YCrCb color space.
        
        Range: 0.0 (pitch black) to 255.0 (pure white).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0
        # Fast Y channel extraction: Y = 0.299*R + 0.587*G + 0.114*B
        # Using cv2.cvtColor to YCrCb or direct gray conversion
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @classmethod
    def is_low_light(cls, frame_bgr: np.ndarray, threshold: float = 45.0) -> bool:
        """Return True if frame mean luminance falls below threshold (night / tunnel)."""
        return cls.estimate_mean_luminance(frame_bgr) < threshold


class LowLightPreprocessor:
    """Configurable illumination preprocessor for ablation benchmarking (N0 - N3)."""

    def __init__(
        self,
        mode: Literal["N0", "N1", "N2", "N3", "N3_clahe"] = "N0",
        low_light_threshold: float = 45.0,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        gamma: float = 0.65,
    ):
        self.mode = mode
        self.low_light_threshold = low_light_threshold
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.gamma = gamma

        # Initialize CLAHE operator
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )

        # Precompute 256-element Gamma Lookup Table (LUT) for zero-overhead inference
        inv_gamma = self.gamma
        self.gamma_lut = np.array([
            np.clip(pow(i / 255.0, inv_gamma) * 255.0, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)

    def apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the lightness channel in LAB color space."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply gamma correction via precomputed 256-value LUT."""
        return cv2.LUT(frame_bgr, self.gamma_lut)

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process input frame according to current ablation mode.
        
        Returns:
            Tuple of (processed_frame, telemetry_dict)
        """
        t0 = time.perf_counter()
        mean_y = IlluminationEstimator.estimate_mean_luminance(frame_bgr)
        applied = False
        processed = frame_bgr

        if self.mode == "N0":
            # Baseline: untouched
            processed = frame_bgr
            applied = False
        elif self.mode == "N1":
            # CLAHE unconditional
            processed = self.apply_clahe(frame_bgr)
            applied = True
        elif self.mode == "N2":
            # Gamma unconditional
            processed = self.apply_gamma(frame_bgr)
            applied = True
        elif self.mode == "N3":
            # Adaptive Gamma: trigger only if scene is genuinely low-light
            if mean_y < self.low_light_threshold:
                processed = self.apply_gamma(frame_bgr)
                applied = True
            else:
                processed = frame_bgr
                applied = False
        elif self.mode == "N3_clahe":
            # Adaptive CLAHE: trigger only if scene is genuinely low-light
            if mean_y < self.low_light_threshold:
                processed = self.apply_clahe(frame_bgr)
                applied = True
            else:
                processed = frame_bgr
                applied = False
        else:
            raise ValueError(f"Unknown low-light preprocessing mode: {self.mode}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "mode": self.mode,
            "mean_luminance": mean_y,
            "applied": applied,
            "latency_ms": elapsed_ms,
        }
        return processed, telemetry
