"""Unit tests for Heading and Directional Cosine Calculation."""

import pytest
import numpy as np
from road_surveillance.geometry.spatial_metrics import calculate_directional_cosine


def test_directional_cosine_aligned():
    # Both heading North
    v_head = (0.0, 1.0)
    v_lane = (0.0, 1.0)
    cos_val = calculate_directional_cosine(v_head, v_lane)
    assert pytest.approx(cos_val) == 1.0


def test_directional_cosine_opposing_wrong_way():
    # Heading South while lane is North
    v_head = (0.0, -1.0)
    v_lane = (0.0, 1.0)
    cos_val = calculate_directional_cosine(v_head, v_lane)
    assert pytest.approx(cos_val) == -1.0


def test_directional_cosine_perpendicular():
    # Heading East while lane is North
    v_head = (1.0, 0.0)
    v_lane = (0.0, 1.0)
    cos_val = calculate_directional_cosine(v_head, v_lane)
    assert pytest.approx(cos_val, abs=1e-5) == 0.0


def test_directional_cosine_diagonal():
    v_head = (1.0, 1.0)
    v_lane = (0.0, 1.0)
    cos_val = calculate_directional_cosine(v_head, v_lane)
    # 45 deg -> cos = 1 / sqrt(2) ≈ 0.7071
    assert pytest.approx(cos_val, rel=1e-3) == float(1.0 / np.sqrt(2))
