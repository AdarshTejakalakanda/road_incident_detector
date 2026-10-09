"""Unit tests for Iteration J: Perception Recovery and ByteTrack Deformation Coasting."""

import pytest
import numpy as np

from road_surveillance.tracking.track_state import Track, TrackletState
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch


def test_track_label_consistency_smoothing():
    tr = Track(track_id=1, class_id=2, class_label="car")
    # Vehicle detected for 5 frames
    for i in range(5):
        tr.update(bbox_xywh=(100, 100, 50, 40), confidence=0.75, pts_ns=i * 33333333, label="car")
    assert tr.class_label == "car"

    # Glare causes 2 frames of non-vehicle noise ('clock' and 'bottle')
    tr.update(bbox_xywh=(102, 101, 50, 40), confidence=0.28, pts_ns=5 * 33333333, label="clock")
    tr.update(bbox_xywh=(103, 102, 50, 40), confidence=0.26, pts_ns=6 * 33333333, label="bottle")

    # Majority label smoothing ensures it stays 'car'
    assert tr.class_label == "car"
    assert tr.get_effective_label() == "car"


def test_train_to_truck_vocabulary_remapping():
    tracker = ByteTrackTracker(track_activation_threshold=0.20)
    assert tracker.tracker.det_thresh == 0.20

    # Detection batch with 'train'
    batch = DetectionBatch(
        video_id="v1",
        frame_idx=1,
        pts_ns=33333333,
        boxes=np.array([[200, 200, 400, 300]], dtype=np.float32),
        confidences=np.array([0.70], dtype=np.float32),
        class_ids=np.array([6], dtype=np.int32),  # COCO 6 = train
        class_labels=["train"],
    )

    tracklets, tracks = tracker.update(batch)
    assert len(tracklets) == 1
    # Check that 'train' was remapped to 'truck'
    assert tracklets[0].class_label == "truck"
    tid = tracklets[0].track_id
    assert tracks[tid].class_label == "truck"


def test_collision_deformation_coasting():
    tracker = ByteTrackTracker(track_activation_threshold=0.20)

    # Establish mature track across 4 frames
    for f in range(4):
        batch = DetectionBatch(
            video_id="v1",
            frame_idx=f,
            pts_ns=f * 33333333,
            boxes=np.array([[100 + f * 5, 200, 150 + f * 5, 250]], dtype=np.float32),
            confidences=np.array([0.80], dtype=np.float32),
            class_ids=np.array([2], dtype=np.int32),
            class_labels=["car"],
        )
        tracklets, tracks = tracker.update(batch)
        assert len(tracklets) == 1
        assert tracklets[0].track_status == "active"

    # Frame 4: Crash happens, vehicle crumples, YOLO confidence collapses (0 detections)
    empty_batch = DetectionBatch(
        video_id="v1",
        frame_idx=4,
        pts_ns=4 * 33333333,
        boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        class_ids=np.empty((0,), dtype=np.int32),
        class_labels=[],
    )

    coasting_tracklets, tracks = tracker.update(empty_batch)
    # Tracklet should be coasted, NOT immediately dropped
    assert len(coasting_tracklets) == 1
    assert coasting_tracklets[0].track_status == "coasting"
    assert tracks[coasting_tracklets[0].track_id].status == "coasting"
    assert tracks[coasting_tracklets[0].track_id].coasting_count == 1
