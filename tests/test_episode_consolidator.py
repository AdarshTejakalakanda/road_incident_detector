"""Unit tests for Episode Consolidation and Progression Tracking (Iteration P)."""

import pytest
from modules.episode_consolidator import (
    calculate_candidate_saliency,
    consolidate_candidates_into_episodes,
    ConsolidatedEpisode,
)


def test_calculate_candidate_saliency():
    # Multi-vehicle candidate with high decel and IoU
    cand_high = {
        "event_type": "collision_candidate",
        "frame": 50,
        "confidence": 0.90,
        "signals": {
            "accel_1_ms2": -12.5,
            "bbox_iou": 0.35,
            "ground_distance_m": 0.8,
        },
    }
    score_high = calculate_candidate_saliency(cand_high)

    # Mild anomaly with low kinematics
    cand_low = {
        "event_type": "single_vehicle_anomaly",
        "frame": 50,
        "confidence": 0.50,
        "signals": {
            "accel_1_ms2": -1.0,
            "bbox_iou": 0.0,
            "ground_distance_m": 5.0,
        },
    }
    score_low = calculate_candidate_saliency(cand_low)

    assert score_high > score_low
    assert score_high > 50.0


def test_consolidate_candidates_into_episodes_cooldown_and_tracks():
    # 5 raw candidate proposals across frames 50-54 for tracks [1, 2]
    cands = [
        {
            "frame": 50,
            "timestamp_s": 5.0,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.80,
            "signals": {"accel_1_ms2": -5.0, "bbox_iou": 0.10, "ground_distance_m": 1.5},
        },
        {
            "frame": 51,
            "timestamp_s": 5.1,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.95,
            "signals": {"accel_1_ms2": -14.0, "bbox_iou": 0.30, "ground_distance_m": 0.5},
        },
        {
            "frame": 52,
            "timestamp_s": 5.2,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -8.0, "bbox_iou": 0.20, "ground_distance_m": 0.9},
        },
        # Separate candidate 20 seconds later (frame 300, ts=25.0s) on different tracks [3, 4]
        {
            "frame": 300,
            "timestamp_s": 25.0,
            "event_type": "single_vehicle_anomaly",
            "track_ids": [3],
            "confidence": 0.70,
            "signals": {"accel_1_ms2": -6.0, "composite_score": 0.65},
        },
    ]

    episodes = consolidate_candidates_into_episodes(
        raw_candidates=cands,
        video_id="test_clip",
        cooldown_seconds=10.0,
        gt_frame=51,
        temporal_tolerance=45,
    )

    # 4 raw proposals should cluster into exactly 2 consolidated episodes
    assert len(episodes) == 2

    ep1 = episodes[0]
    assert ep1.start_frame == 50
    assert ep1.end_frame == 52
    assert len(ep1.raw_candidates) == 3
    # Peak candidate should be frame 51 (highest decel -14.0 and highest IoU)
    assert ep1.peak_frame == 51
    assert ep1.is_in_gt_window is True

    ep2 = episodes[1]
    assert ep2.start_frame == 300
    assert ep2.peak_frame == 300
    assert ep2.is_in_gt_window is False


def test_disjoint_actor_pairs_cannot_merge():
    """Verify disjoint actor pairs {7, 14} and {10, 22} create separate episodes even within dt < 1.0s."""
    cands = [
        {
            "frame": 128,
            "timestamp_s": 4.267,
            "event_type": "collision_candidate",
            "track_ids": [7, 14],
            "confidence": 0.90,
            "signals": {"accel_1_ms2": -15.0, "bbox_iou": 0.31, "ground_distance_m": 1.2},
            "ground_coordinates": (15.0, 20.0),
        },
        {
            "frame": 140,
            "timestamp_s": 4.667,
            "event_type": "collision_candidate",
            "track_ids": [10, 22],
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -12.0, "bbox_iou": 0.20, "ground_distance_m": 1.5},
            "ground_coordinates": (16.0, 21.0),
        },
    ]

    episodes = consolidate_candidates_into_episodes(cands, video_id="test_clip")
    assert len(episodes) == 2
    assert episodes[0].primary_actor_pair == (7, 14)
    assert episodes[1].primary_actor_pair == (10, 22)


def test_transitive_chaining_prevented():
    """Verify sequence {1, 2} -> {2, 3} -> {3, 4} does NOT merge {1, 2} with {3, 4}."""
    cands = [
        {
            "frame": 100,
            "timestamp_s": 3.33,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.90,
            "signals": {"accel_1_ms2": -10.0, "bbox_iou": 0.25},
            "ground_coordinates": (10.0, 10.0),
        },
        {
            "frame": 110,
            "timestamp_s": 3.66,
            "event_type": "collision_candidate",
            "track_ids": [2, 3],
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -8.0, "bbox_iou": 0.20},
            "ground_coordinates": (11.0, 10.0),  # within 4m of primary impact
        },
        {
            "frame": 120,
            "timestamp_s": 4.00,
            "event_type": "collision_candidate",
            "track_ids": [3, 4],  # Disjoint from primary {1, 2}
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -9.0, "bbox_iou": 0.22},
            "ground_coordinates": (11.5, 10.0),
        },
    ]

    episodes = consolidate_candidates_into_episodes(cands, video_id="test_clip")
    assert len(episodes) == 2
    # Ep 1 should contain candidates for {1, 2} and secondary {2, 3}
    assert set(episodes[0].primary_actor_pair) == {1, 2}
    assert (2, 3) in episodes[0].secondary_actor_pairs
    assert len(episodes[0].raw_candidates) == 2
    # Ep 2 must be created for {3, 4} because {3, 4} does not intersect primary pair {1, 2}
    assert set(episodes[1].primary_actor_pair) == {3, 4}
    assert len(episodes[1].raw_candidates) == 1


def test_same_actor_pair_continuity():
    """Verify successive candidates for {1, 2} within dt <= 1.8s merge."""
    cands = [
        {
            "frame": 50,
            "timestamp_s": 1.0,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.80,
            "signals": {"accel_1_ms2": -5.0},
        },
        {
            "frame": 65,
            "timestamp_s": 1.5,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.95,
            "signals": {"accel_1_ms2": -15.0},
        },
    ]

    episodes = consolidate_candidates_into_episodes(cands, video_id="test_clip")
    assert len(episodes) == 1
    assert len(episodes[0].raw_candidates) == 2
    assert episodes[0].primary_actor_pair == (1, 2)


def test_pileup_chain_reaction_admitted():
    """Verify primary collision {1, 2} and secondary {2, 3} within 4m merge with secondary pair recorded."""
    cands = [
        {
            "frame": 50,
            "timestamp_s": 1.0,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -10.0},
            "ground_coordinates": (10.0, 10.0),
        },
        {
            "frame": 65,
            "timestamp_s": 1.5,
            "event_type": "collision_candidate",
            "track_ids": [2, 3],
            "confidence": 0.80,
            "signals": {"accel_1_ms2": -6.0},
            "ground_coordinates": (12.0, 10.0),  # dist = 2.0m <= 4.0m
        },
    ]

    episodes = consolidate_candidates_into_episodes(cands, video_id="test_clip")
    assert len(episodes) == 1
    assert episodes[0].primary_actor_pair == (1, 2)
    assert (2, 3) in episodes[0].secondary_actor_pairs


def test_pileup_chain_reaction_rejected_at_distance():
    """Verify secondary candidate {2, 3} at distance > 4.0m is not merged."""
    cands = [
        {
            "frame": 50,
            "timestamp_s": 1.0,
            "event_type": "collision_candidate",
            "track_ids": [1, 2],
            "confidence": 0.85,
            "signals": {"accel_1_ms2": -10.0},
            "ground_coordinates": (10.0, 10.0),
        },
        {
            "frame": 65,
            "timestamp_s": 1.5,
            "event_type": "collision_candidate",
            "track_ids": [2, 3],
            "confidence": 0.80,
            "signals": {"accel_1_ms2": -6.0},
            "ground_coordinates": (20.0, 10.0),  # dist = 10.0m > 4.0m
        },
    ]

    episodes = consolidate_candidates_into_episodes(cands, video_id="test_clip")
    assert len(episodes) == 2

