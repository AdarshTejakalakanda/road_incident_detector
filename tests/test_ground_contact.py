"""Unit tests for Ground-Contact Point Calculation."""

import pytest
from road_surveillance.tracking.track_state import Track


def test_ground_contact_exact_formula():
    # bbox format: (bx, by, bw, bh)
    bbox = (100.0, 50.0, 40.0, 60.0)
    gc_x, gc_y = Track.calculate_ground_contact(bbox)

    # Expected: x = 100 + 40/2 = 120, y = 50 + 60 = 110
    assert gc_x == 120.0
    assert gc_y == 110.0


def test_ground_contact_zero_origin():
    bbox = (0.0, 0.0, 200.0, 150.0)
    gc_x, gc_y = Track.calculate_ground_contact(bbox)

    assert gc_x == 100.0
    assert gc_y == 150.0


def test_ground_contact_floating_point():
    bbox = (10.5, 20.25, 30.5, 40.75)
    gc_x, gc_y = Track.calculate_ground_contact(bbox)

    assert pytest.approx(gc_x) == 10.5 + 30.5 / 2.0
    assert pytest.approx(gc_y) == 20.25 + 40.75
