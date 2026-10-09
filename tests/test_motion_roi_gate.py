"""Unit tests for modules/motion_roi_gate.py per Patch N.2 specification."""

from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

from modules.motion_roi_gate import (
    MotionROIGate,
    compute_box_iou,
    extract_roadway_motion_clusters,
)


def test_box_iou_computation():
    b1 = (10.0, 10.0, 50.0, 50.0)
    b2 = (10.0, 10.0, 50.0, 50.0)
    assert compute_box_iou(b1, b2) == pytest.approx(1.0)

    b3 = (100.0, 100.0, 150.0, 150.0)
    assert compute_box_iou(b1, b3) == pytest.approx(0.0)

    b4 = (30.0, 10.0, 70.0, 50.0)  # 50% overlap
    iou = compute_box_iou(b1, b4)
    assert 0.30 < iou < 0.40


def test_extract_roadway_motion_clusters_synthetic():
    h, w = 360, 640
    frame_prev = np.zeros((h, w, 3), dtype=np.uint8)
    frame_curr = np.zeros((h, w, 3), dtype=np.uint8)

    # Roadway mask: full frame
    roadway_mask = np.full((h, w), 255, dtype=np.uint8)

    # In frame_curr, draw a moving white rectangle of area ~500 px^2 (e.g. 25x20)
    cv2.rectangle(frame_curr, (200, 150), (225, 170), (255, 255, 255), -1)

    clusters = extract_roadway_motion_clusters(
        frame_curr=frame_curr,
        frame_prev=frame_prev,
        roadway_mask=roadway_mask,
        min_cluster_area=100,
        max_cluster_area=2000,
    )

    assert len(clusters) == 1
    x1, y1, x2, y2, area = clusters[0]
    assert area >= 100
    # Must have 50% spatial padding around [200, 150, 225, 170]
    assert x1 < 200 and y1 < 150
    assert x2 > 225 and y2 > 170


def test_motion_roi_gate_lifecycle_and_max_crop_budget():
    gate = MotionROIGate(
        min_cluster_area=100,
        max_cluster_area=2000,
        persistence_threshold=4,
        max_crops_per_frame=1,
    )

    h, w = 360, 640
    road_poly = [(0, h), (w, h), (int(w * 0.75), int(h * 0.35)), (int(w * 0.25), int(h * 0.35))]

    # Mock YOLO model
    mock_model = MagicMock()
    mock_res = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = np.array([[50.0, 50.0, 100.0, 100.0]], dtype=np.float32)
    mock_box.conf = np.array([0.45], dtype=np.float32)
    mock_box.cls = np.array([2], dtype=np.float32)  # car
    mock_res.boxes = MagicMock()
    mock_res.boxes.__len__.return_value = 1
    mock_res.boxes.xyxy = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=mock_box.xyxy))))
    mock_res.boxes.conf = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=mock_box.conf))))
    mock_res.boxes.cls = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=mock_box.cls))))
    mock_model.predict.return_value = [mock_res]
    mock_model.names = {2: "car"}

    frame_base = np.zeros((h, w, 3), dtype=np.uint8)

    # Feed frames 1 through 3: cluster age increases, but < 4 -> no crops
    for f in range(1, 4):
        frame = frame_base.copy()
        # Draw moving patch in roadway
        x_pos = 300 + f * 4
        cv2.rectangle(frame, (x_pos, 200), (x_pos + 30, 220), (255, 255, 255), -1)
        recs, tel = gate.process_and_recover(
            frame_curr=frame,
            frame_idx=f,
            primary_raw_detections=[],
            roadway_polygon=road_poly,
            yolo_model=mock_model,
        )
        assert len(recs) == 0
        assert tel["roi_crops_evaluated"] == 0

    # Frame 4: cluster age reaches 4 (or persistence threshold) -> exactly 1 crop evaluated!
    frame_4 = frame_base.copy()
    x_pos = 300 + 4 * 4
    cv2.rectangle(frame_4, (x_pos, 200), (x_pos + 30, 220), (255, 255, 255), -1)
    recs, tel = gate.process_and_recover(
        frame_curr=frame_4,
        frame_idx=4,
        primary_raw_detections=[],
        roadway_polygon=road_poly,
        yolo_model=mock_model,
    )
    assert tel["roi_crops_evaluated"] == 1
    assert len(recs) == 1
    rec = recs[0]
    assert rec["class_name"] == "candidate_vehicle_recovered"
    assert rec["original_class_name"] == "car"
    assert rec["recovery_reason"] == "motion_guided_roi_zoom"
    assert rec["audit_reclassified"] is True
