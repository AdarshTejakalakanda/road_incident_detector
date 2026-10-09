"""Frame Quality & Lighting Classifier and Illumination-Gated Preprocessing Module.

Iteration K.2: Illumination-Gated Preprocessing Pipeline
Objective:
Evaluate whether targeted low-light enhancement (Gamma adjustment, CLAHE on luminance,
and adaptive routing) pulls CAT_2 detections above threshold and sharpens vehicle edges
to convert CAT_4 into valid IoU matches, without inducing false positives or adding
> 2 ms processing overhead.

Metrics:
- Mean Luminance (mu_V): (1/N) * sum(V(x, y)) in HSV space
- Glare Saturation Ratio (R_glare): sum[V(x, y) > 245] / N

Classification Enum:
- DAY_NORMAL: mu_V >= 65 and R_glare < 0.05 (or mu_V >= 65)
- LOW_LIGHT: mu_V < 65 and R_glare < 0.03
- NIGHT_HIGH_GLARE: mu_V < 65 and R_glare >= 0.03
"""

import time
from enum import Enum
from typing import Dict, Tuple, Any, Optional
import numpy as np
import cv2


class LightingCondition(str, Enum):
    """Scene illumination classification enum."""
    DAY_NORMAL = "DAY_NORMAL"
    LOW_LIGHT = "LOW_LIGHT"
    NIGHT_HIGH_GLARE = "NIGHT_HIGH_GLARE"


def compute_gamma_lut(gamma: float) -> np.ndarray:
    """Precompute 256-element Gamma Lookup Table for zero-overhead inference."""
    lut = np.array([
        np.clip(pow(i / 255.0, gamma) * 255.0, 0.0, 255.0)
        for i in range(256)
    ], dtype=np.uint8)
    return lut


class IlluminationClassifier:
    """Zero-cost (< 1.5 ms) illumination classification module running before detector inference."""

    def __init__(
        self,
        luminance_threshold: float = 65.0,
        glare_threshold_night: float = 0.03,
        glare_threshold_day: float = 0.05,
        glare_pixel_val: int = 245,
    ):
        self.luminance_threshold = luminance_threshold
        self.glare_threshold_night = glare_threshold_night
        self.glare_threshold_day = glare_threshold_day
        self.glare_pixel_val = glare_pixel_val

    def extract_v_channel(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract HSV Value (luminance) channel fast: V(x, y) = max(B, G, R)."""
        b, g, r = cv2.split(frame_bgr)
        return cv2.max(cv2.max(b, g), r)

    def classify(self, frame_bgr: np.ndarray) -> Tuple[LightingCondition, float, float, float]:
        """Classify illumination condition and return (condition, mu_V, R_glare, elapsed_ms)."""
        t0 = time.perf_counter()
        if frame_bgr is None or frame_bgr.size == 0:
            return LightingCondition.DAY_NORMAL, 0.0, 0.0, 0.0

        v = self.extract_v_channel(frame_bgr)
        total_pixels = v.size

        mu_v = float(np.mean(v))
        glare_count = int(np.count_nonzero(v > self.glare_pixel_val))
        r_glare = float(glare_count / total_pixels) if total_pixels > 0 else 0.0

        if mu_v >= self.luminance_threshold:
            condition = LightingCondition.DAY_NORMAL
        elif r_glare >= self.glare_threshold_night:
            condition = LightingCondition.NIGHT_HIGH_GLARE
        else:
            condition = LightingCondition.LOW_LIGHT

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return condition, round(mu_v, 2), round(r_glare, 4), elapsed_ms


class GammaPreprocessor:
    """Fast Power-Law Gamma Correction via 256-element LUT."""

    def __init__(self, gamma: float = 0.65):
        self.gamma = gamma
        self.lut = compute_gamma_lut(gamma)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.LUT(frame_bgr, self.lut)


class ClahePreprocessor:
    """Selective CLAHE on L-channel in LAB color space."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        # Warmup OpenCV internal kernels
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        lab = cv2.cvtColor(dummy, cv2.COLOR_BGR2LAB)
        self.clahe.apply(lab[:, :, 0])

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


class LightingGate:
    """Illumination-gated preprocessing pipeline controller (K2.0 - K2.3)."""

    def __init__(
        self,
        mode: str = "raw",  # "raw" (K2.0), "gamma" (K2.1), "clahe" (K2.2), "adaptive" (K2.3)
        luminance_threshold: float = 65.0,
        glare_threshold_night: float = 0.03,
    ):
        self.mode = mode.lower()
        self.classifier = IlluminationClassifier(
            luminance_threshold=luminance_threshold,
            glare_threshold_night=glare_threshold_night,
        )

        # Precompute preprocessors
        self.gamma_065 = GammaPreprocessor(gamma=0.65)
        self.gamma_080 = GammaPreprocessor(gamma=0.80)
        self.clahe_20 = ClahePreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
        self.clahe_15 = ClahePreprocessor(clip_limit=1.5, tile_grid_size=(8, 8))

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame according to configured mode.

        Returns:
            Tuple of (enhanced_frame, telemetry_dict)
        """
        t0 = time.perf_counter()

        # Step 1: Illumination classification
        condition, mu_v, r_glare, t_class_ms = self.classifier.classify(frame_bgr)

        # Step 2: Preprocessing routing
        applied_ops = []
        out_frame = frame_bgr

        if self.mode in ("raw", "k2.0", "baseline"):
            out_frame = frame_bgr
            applied_ops.append("none")

        elif self.mode in ("gamma", "k2.1"):
            out_frame = self.gamma_065.apply(frame_bgr)
            applied_ops.append("gamma_0.65")

        elif self.mode in ("clahe", "k2.2"):
            out_frame = self.clahe_20.apply(frame_bgr)
            applied_ops.append("clahe_2.0")

        elif self.mode in ("adaptive", "k2.3"):
            if condition == LightingCondition.DAY_NORMAL:
                out_frame = frame_bgr
                applied_ops.append("passthrough_raw")
            elif condition == LightingCondition.LOW_LIGHT:
                out_frame = self.gamma_065.apply(frame_bgr)
                applied_ops.append("gamma_0.65")
            elif condition == LightingCondition.NIGHT_HIGH_GLARE:
                clahe_frame = self.clahe_15.apply(frame_bgr)
                out_frame = self.gamma_080.apply(clahe_frame)
                applied_ops.append("clahe_1.5+gamma_0.80")
            else:
                out_frame = frame_bgr
                applied_ops.append("fallback_raw")
        else:
            raise ValueError(f"Unknown preprocessing mode: {self.mode}")

        total_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "mode": self.mode,
            "lighting_condition": condition.value,
            "mean_luminance": mu_v,
            "glare_ratio": r_glare,
            "classifier_latency_ms": round(t_class_ms, 3),
            "total_preprocessing_latency_ms": round(total_elapsed_ms, 3),
            "applied_operations": applied_ops,
        }

        return out_frame, telemetry
