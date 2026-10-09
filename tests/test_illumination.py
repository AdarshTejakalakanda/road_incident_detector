"""Unit tests for low-light preprocessing and illumination estimator."""

import pytest
import numpy as np
from road_surveillance.preprocessing.illumination import IlluminationEstimator, LowLightPreprocessor


def test_illumination_estimator_detects_dark_and_bright():
    dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)

    assert IlluminationEstimator.is_low_light(dark_frame, threshold=45.0) is True
    assert IlluminationEstimator.is_low_light(bright_frame, threshold=45.0) is False

    lum_dark = IlluminationEstimator.estimate_mean_luminance(dark_frame)
    lum_bright = IlluminationEstimator.estimate_mean_luminance(bright_frame)
    assert 19.0 <= lum_dark <= 21.0
    assert 179.0 <= lum_bright <= 181.0


def test_low_light_preprocessor_n0_baseline():
    preprocessor = LowLightPreprocessor(mode="N0")
    frame = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    processed, telem = preprocessor.process(frame)

    assert telem["mode"] == "N0"
    assert telem["applied"] is False
    assert np.array_equal(processed, frame)


def test_low_light_preprocessor_n1_clahe():
    preprocessor = LowLightPreprocessor(mode="N1", clahe_clip_limit=2.0)
    frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    # Add a small gradient so CLAHE has contrast to redistribute
    frame[20:80, 20:80] = 50

    processed, telem = preprocessor.process(frame)
    assert telem["mode"] == "N1"
    assert telem["applied"] is True
    assert processed.shape == frame.shape
    assert processed.dtype == np.uint8


def test_low_light_preprocessor_n2_gamma():
    preprocessor = LowLightPreprocessor(mode="N2", gamma=0.65)
    # Dark pixel value 30
    frame = np.full((50, 50, 3), 30, dtype=np.uint8)
    processed, telem = preprocessor.process(frame)

    assert telem["mode"] == "N2"
    assert telem["applied"] is True
    # With gamma=0.65, 255 * (30/255)^0.65 is ~63 > 30
    assert np.all(processed > frame)


def test_low_light_preprocessor_n3_adaptive():
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=45.0, gamma=0.65)

    # 1. Dark frame should be enhanced
    dark_frame = np.full((50, 50, 3), 25, dtype=np.uint8)
    p_dark, telem_dark = preprocessor.process(dark_frame)
    assert telem_dark["applied"] is True
    assert np.all(p_dark > dark_frame)

    # 2. Bright daytime frame should be left completely untouched
    day_frame = np.full((50, 50, 3), 120, dtype=np.uint8)
    p_day, telem_day = preprocessor.process(day_frame)
    assert telem_day["applied"] is False
    assert np.array_equal(p_day, day_frame)
