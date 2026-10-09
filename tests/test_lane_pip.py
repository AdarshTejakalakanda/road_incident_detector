"""Unit tests for Lane Point-in-Polygon (PIP) Containment."""

import pytest
from road_surveillance.calibration.lane_manager import Lane, LaneManager


def test_lane_point_in_polygon():
    # Square lane from (-5, 0) to (5, 50)
    polygon = [(-5.0, 0.0), (5.0, 0.0), (5.0, 50.0), (-5.0, 50.0)]
    lane = Lane(lane_id="L1", name="Main", is_active=True, permitted_direction=(0.0, 1.0), polygon_ground=polygon)

    # Point clearly inside
    assert lane.contains_ground_point(0.0, 25.0) is True

    # Point clearly outside (left)
    assert lane.contains_ground_point(-10.0, 25.0) is False

    # Point clearly outside (beyond y=50)
    assert lane.contains_ground_point(0.0, 60.0) is False

    # Point on boundary
    assert lane.contains_ground_point(5.0, 25.0) is True


def test_lane_manager_resolution():
    poly1 = [(-10.0, 0.0), (0.0, 0.0), (0.0, 50.0), (-10.0, 50.0)]
    poly2 = [(0.0, 0.0), (10.0, 0.0), (10.0, 50.0), (0.0, 50.0)]

    lane1 = Lane("L_left", "Left", True, (0, 1), poly1)
    lane2 = Lane("L_right", "Right", False, (0, 1), poly2)

    lm = LaneManager([lane1, lane2])

    # In lane 1
    found_lane = lm.get_lane_for_ground_point(-5.0, 20.0)
    assert found_lane is not None
    assert found_lane.lane_id == "L_left"
    assert lm.is_in_active_lane(-5.0, 20.0) is True

    # In lane 2 (inactive)
    assert lm.is_in_active_lane(5.0, 20.0) is False
