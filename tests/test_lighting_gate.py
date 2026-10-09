"""Unit tests for modules/lighting_gate.py (Iteration K.2)."""

import pytest
import numpy as np
import cv2
from modules.lighting_gate import (
    LightingCondition,
    IlluminationClassifier,
    GammaPreprocessor,
    ClahePreprocessor,
    LightingGate,
    compute_gamma_lut,
)


def test_lighting_classifier_day_normal():
    clf = IlluminationClassifier(luminance_threshold=65.0, glare_threshold_night=0.03)
    # Bright daylight frame (V = 100)
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    cond, mu_v, r_glare, elapsed_ms = clf.classify(frame)

    assert cond == LightingCondition.DAY_NORMAL
    assert mu_v == 100.0
    assert r_glare == 0.0
    assert elapsed_ms < 5.0


def test_lighting_classifier_low_light():
    clf = IlluminationClassifier(luminance_threshold=65.0, glare_threshold_night=0.03)
    # Dark frame (V = 35) without glare
    frame = np.full((120, 160, 3), 35, dtype=np.uint8)
    cond, mu_v, r_glare, elapsed_ms = clf.classify(frame)

    assert cond == LightingCondition.LOW_LIGHT
    assert mu_v == 35.0
    assert r_glare == 0.0


def test_lighting_classifier_night_high_glare():
    clf = IlluminationClassifier(luminance_threshold=65.0, glare_threshold_night=0.03)
    # Dark frame with 5% high glare pixels (> 245)
    frame = np.full((100, 100, 3), 30, dtype=np.uint8)
    frame[:10, :50, :] = 250  # 500 / 10000 = 5% glare

    cond, mu_v, r_glare, elapsed_ms = clf.classify(frame)
    assert cond == LightingCondition.NIGHT_HIGH_GLARE
    assert mu_v < 65.0
    assert r_glare >= 0.03


def test_gamma_lut_mathematical_precision():
    lut = compute_gamma_lut(0.65)
    assert len(lut) == 256
    assert lut[0] == 0
    assert lut[255] == 255
    # Mid-tone test: 64/255 = 0.25098 -> (0.25098)^0.65 = 0.4068 -> 103.7 -> 104
    assert abs(int(lut[64]) - 104) <= 1

    preproc = GammaPreprocessor(gamma=0.65)
    img = np.array([[[64, 128, 200]]], dtype=np.uint8)
    out = preproc.apply(img)
    assert out.shape == img.shape
    assert out[0, 0, 0] == lut[64]


def test_clahe_preprocessor():
    preproc = ClahePreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
    img = np.random.randint(20, 80, (120, 160, 3), dtype=np.uint8)
    out = preproc.apply(img)

    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_lighting_gate_adaptive_routing():
    gate = LightingGate(mode="adaptive")

    # Day normal -> passthrough raw
    day_frame = np.full((60, 80, 3), 120, dtype=np.uint8)
    out_day, tel_day = gate.process(day_frame)
    assert tel_day["lighting_condition"] == LightingCondition.DAY_NORMAL.value
    assert "passthrough_raw" in tel_day["applied_operations"]
    assert np.array_equal(out_day, day_frame)

    # Low light -> gamma 0.65
    low_frame = np.full((60, 80, 3), 30, dtype=np.uint8)
    out_low, tel_low = gate.process(low_frame)
    assert tel_low["lighting_condition"] == LightingCondition.LOW_LIGHT.value
    assert "gamma_0.65" in tel_low["applied_operations"]
    assert not np.array_equal(out_low, low_frame)

    # Night high glare -> CLAHE + gamma 0.80
    glare_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    glare_frame[:10, :50, :] = 250
    out_glare, tel_glare = gate.process(glare_frame)
    assert tel_glare["lighting_condition"] == LightingCondition.NIGHT_HIGH_GLARE.value
    assert "clahe_1.5+gamma_0.80" in tel_glare["applied_operations"]
    assert out_glare.shape == glare_frame.shape
