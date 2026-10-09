"""Unit tests for RobustSurveillanceTracker."""

import pytest
import numpy as np

from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.detection.detector_interface import DetectionBatch


def test_entrant_track_survival_with_low_confidence_frame2():
    """Verify that a track initiated at frame 1 survives when frame 2 has low confidence."""
    tracker = RobustSurveillanceTracker(track_activation_threshold=0.25, low_confidence_floor=0.10)

    # Frame 1: Vehicle appears with high confidence (0.40)
    batch1 = DetectionBatch(
        video_id="vid",
        frame_idx=1,
        pts_ns=33333333,
        boxes=np.array([[100, 100, 150, 150]], dtype=np.float32),
        confidences=np.array([0.40], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tls1, tracks1 = tracker.update(batch1)
    assert len(tracks1) == 1
    assert 1 in tracks1
    assert tracks1[1].track_age == 1

    # Frame 2: Vehicle confidence dips to 0.18 (< 0.25 threshold, e.g. shadow or glare)
    batch2 = DetectionBatch(
        video_id="vid",
        frame_idx=2,
        pts_ns=66666666,
        boxes=np.array([[102, 101, 152, 151]], dtype=np.float32),
        confidences=np.array([0.18], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tls2, tracks2 = tracker.update(batch2)

    # In supervision.ByteTrack, this was deleted! In RobustSurveillanceTracker, it survives and matures!
    assert len(tracks2) == 1
    assert 1 in tracks2
    assert tracks2[1].track_age == 2
    assert tracks2[1].status == "active"
    assert len(tls2) == 1


def test_low_confidence_recovery_during_collision():
    """Verify Stage 2 association matches low-confidence crumpled vehicle during crash."""
    tracker = RobustSurveillanceTracker(track_activation_threshold=0.25, low_confidence_floor=0.10)

    # Mature track across 3 frames
    for f in range(3):
        b = DetectionBatch(
            video_id="vid",
            frame_idx=f,
            pts_ns=f * 33333333,
            boxes=np.array([[200 + f * 4, 300, 260 + f * 4, 350]], dtype=np.float32),
            confidences=np.array([0.75], dtype=np.float32),
            class_ids=np.array([2], dtype=np.int32),
            class_labels=["car"],
        )
        tracker.update(b)

    assert 1 in tracker.tracks
    assert tracker.tracks[1].track_age == 3

    # Frame 3: Impact! Confidence collapses from 0.75 to 0.14
    crash_batch = DetectionBatch(
        video_id="vid",
        frame_idx=3,
        pts_ns=3 * 33333333,
        boxes=np.array([[214, 301, 274, 351]], dtype=np.float32),
        confidences=np.array([0.14], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tls3, tracks3 = tracker.update(crash_batch)

    assert 1 in tracks3
    assert tracks3[1].status == "active"
    assert tracks3[1].track_age == 4
    assert tracks3[1].detection_confidence == pytest.approx(0.14)


def test_deformation_coasting_on_empty_frame():
    """Verify mature vehicle track enters coasting when occluded or completely undetectable."""
    tracker = RobustSurveillanceTracker(track_activation_threshold=0.20, max_coasting_frames=5)

    # Mature track
    for f in range(3):
        tracker.update(
            DetectionBatch(
                video_id="vid",
                frame_idx=f,
                pts_ns=f * 33333333,
                boxes=np.array([[300 + f * 5, 400, 360 + f * 5, 450]], dtype=np.float32),
                confidences=np.array([0.80], dtype=np.float32),
                class_ids=np.array([2], dtype=np.int32),
                class_labels=["car"],
            )
        )

    # Empty frame (heavy smoke/dust after crash)
    empty_batch = DetectionBatch(
        video_id="vid",
        frame_idx=3,
        pts_ns=3 * 33333333,
        boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        class_ids=np.empty((0,), dtype=np.int32),
        class_labels=[],
    )
    tls, tracks = tracker.update(empty_batch)

    assert 1 in tracks
    assert tracks[1].status == "coasting"
    assert tracks[1].coasting_count == 1
    assert len(tls) == 1
    assert tls[0].track_status == "coasting"
    assert tracks[1].provenance_history[-1] == "PREDICTED"


def test_track_provenance_and_gap_interpolation():
    """Verify that gap <= 6 frames converts PREDICTED frames to INTERPOLATED upon re-detection."""
    tracker = RobustSurveillanceTracker(track_activation_threshold=0.20, max_coasting_frames=8)

    # Frame 0: Real detection
    b0 = DetectionBatch(
        video_id="vid",
        frame_idx=0,
        pts_ns=0,
        boxes=np.array([[100, 100, 150, 150]], dtype=np.float32),
        confidences=np.array([0.80], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tracker.update(b0)
    # Frame 1: Real detection
    b1 = DetectionBatch(
        video_id="vid",
        frame_idx=1,
        pts_ns=33333333,
        boxes=np.array([[105, 100, 155, 150]], dtype=np.float32),
        confidences=np.array([0.80], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tracker.update(b1)
    tr = tracker.tracks[1]
    assert tr.provenance_history == ["REAL", "REAL"]

    # Frames 2, 3: Missing detections (coasting / PREDICTED)
    empty = DetectionBatch(
        video_id="vid",
        frame_idx=2,
        pts_ns=66666666,
        boxes=np.empty((0, 4), dtype=np.float32),
        confidences=np.empty((0,), dtype=np.float32),
        class_ids=np.empty((0,), dtype=np.int32),
        class_labels=[],
    )
    tracker.update(empty)
    empty.frame_idx = 3
    empty.pts_ns = 99999999
    tracker.update(empty)
    assert tr.provenance_history == ["REAL", "REAL", "PREDICTED", "PREDICTED"]

    # Frame 4: Re-detected! Gap = 2 frames (<= 6 frames)
    b4 = DetectionBatch(
        video_id="vid",
        frame_idx=4,
        pts_ns=133333332,
        boxes=np.array([[120, 100, 170, 150]], dtype=np.float32),
        confidences=np.array([0.80], dtype=np.float32),
        class_ids=np.array([2], dtype=np.int32),
        class_labels=["car"],
    )
    tracker.update(b4)

    # Intermediate frames 2 and 3 should now be retroactively tagged "INTERPOLATED"
    assert tr.provenance_history == ["REAL", "REAL", "INTERPOLATED", "INTERPOLATED", "REAL"]
    # Check that interpolated positions lie between frame 1 and frame 4
    p1 = tr.position_history[1][0]
    p2 = tr.position_history[2][0]
    p3 = tr.position_history[3][0]
    p4 = tr.position_history[4][0]
    assert p1 < p2 < p3 < p4


def test_safety_invariant_blocks_collision_on_purely_predicted_tracks():
    """Verify that a collision candidate will NOT trigger if either track lacks a REAL observation near the event."""
    from road_surveillance.events.candidates.collision import CollisionCandidateDetector
    from road_surveillance.calibration.lane_manager import LaneManager
    from road_surveillance.tracking.track_state import Track
    from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence

    detector = CollisionCandidateDetector()
    lm = LaneManager()

    tr1 = Track(1, 2, "car")
    tr2 = Track(2, 2, "car")
    tr1.current_bbox_xywh = (100, 100, 50, 40)
    tr2.current_bbox_xywh = (102, 102, 50, 40)

    # tr1 has REAL detections recently
    tr1.provenance_history = ["REAL", "REAL", "REAL"]
    # tr2 has been lost/coasting for 5 frames without any REAL detection (exceeds window=3)
    tr2.provenance_history = ["REAL", "PREDICTED", "PREDICTED", "PREDICTED", "PREDICTED", "PREDICTED"]

    conv = TrajectoryConvergence(1, 2, distance_m=1.0, is_converging=True, tca_sec=0.1, dca_m=0.2)
    kin1 = TrackKinematics(1, 0, 0, (0, 0), (0, 0), 0.0, 0.0, (-8, 0), -8.0, 0, (1, 0), 0, None, True)
    kin2 = TrackKinematics(2, 0, 0, (1, 0), (0, 0), 0.0, 0.0, (-8, 0), -8.0, 180, (-1, 0), 0, None, True)

    cands = detector.evaluate(0, 0, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)
    # Blocked by Safety Invariant 1!
    assert len(cands) == 0

    # If tr2 gets a REAL observation, it passes the invariant
    tr2.provenance_history.append("REAL")
    cands_valid = detector.evaluate(0, 0, "v1", {1: tr1, 2: tr2}, {1: kin1, 2: kin2}, [conv], lm)
    assert len(cands_valid) == 1

