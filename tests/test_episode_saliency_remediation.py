"""Unit regression test suite for PTS-Native Temporal Saliency Anchor Remediation (Stage T Layer 2).

Validates:
1. Test 1 (test_uarp01_f27_selected_over_f63): Numerical dominance of genuine crash F27 over F63,
   and selection of F27 Track 3 as the episode representative dispatched to Gate 4.
2. Test 2 (test_18_tp_anchors_invariant): 100% (18/18) invariance of primary TP anchor frames
   across historical Post-A/B True Positive episodes.
3. Test 3 (test_zero_penalty_fallback): Verification of zero-penalty fallback when onset is unavailable.
"""

import json
import os
import pytest
from modules.episode_consolidator import (
    calculate_candidate_saliency,
    consolidate_candidates_into_episodes,
    ConsolidatedEpisode,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_uarp01_f27_selected_over_f63():
    """Verify that in UarP8qU1S-c_01, the genuine crash F27 numerically dominates F63

    and is selected as the representative anchor dispatched to Gate 4.
    """
    frame_pts_map = {27: 1.810, 30: 2.011, 63: 4.223}

    # Candidate A: Genuine Barrier Strike / Single Vehicle Anomaly at F27 (Track 3)
    cand_f27 = {
        "frame": 27,
        "timestamp_s": 1.810,
        "event_type": "barrier_strike",
        "track_ids": [3],
        "confidence": 0.90,
        "signals": {
            "metric_accel_ms2": -60.0,
            "composite_score": 0.7171,
            "decel_onset_frame": 27,
            "decel_onset_pts": 1.810,
            "tau_rest_s": 0.201,
            "v_entry_kmh": 43.38,
        },
    }

    # Candidate B: Downstream Cruising Traffic at F63 (Tracks 21 & 22)
    # Relative to incident kinetic onset at Frame 30 (t = 2.011s)
    cand_f63 = {
        "frame": 63,
        "timestamp_s": 4.223,
        "event_type": "collision_candidate",
        "track_ids": [21, 22],
        "confidence": 0.98,
        "signals": {
            "accel_1_ms2": -60.0,
            "bbox_iou": 0.5901,
            "ground_distance_m": 12.75,
            "decel_onset_frame": 30,
            "decel_onset_pts": 2.011,
        },
    }

    # 1. Assert Numerical Dominance: Score(F27) > Score(F63)
    score_f27 = calculate_candidate_saliency(cand_f27, frame_pts_map=frame_pts_map)
    score_f63 = calculate_candidate_saliency(cand_f63, frame_pts_map=frame_pts_map)

    # Detailed accounting:
    # F27: accel (50.0) + IoU (0.0) + dist (0.0) + single (21.51) + type (25.0) + conf (9.0) - penalty (0.0) = 105.51
    # F63: accel (50.0) + IoU (23.60) + dist (0.0) + single (0.0) + type (25.0) + conf (9.8) - penalty (45.0 * 2.212 = 99.54) = 8.86
    assert score_f27 > score_f63, f"Score F27 ({score_f27:.2f}) must exceed Score F63 ({score_f63:.2f})"
    assert score_f27 > 80.0
    assert score_f63 < 60.0

    # 2. Assert Episode Representative Selection: F27 / Track 3 is designated anchor
    episodes = consolidate_candidates_into_episodes(
        raw_candidates=[cand_f27, cand_f63],
        video_id="UarP8qU1S-c_01",
        gt_frame=30,
        frame_pts_map=frame_pts_map,
    )

    in_window_episodes = [ep for ep in episodes if ep.is_in_gt_window]
    assert len(in_window_episodes) >= 1, "At least one in-window episode must exist"

    # Selection rule: Highest peak saliency in-window episode dispatched to Gate 4
    rep_episode = max(in_window_episodes, key=lambda ep: ep.peak_score)
    assert rep_episode.peak_frame == 27, f"Representative peak frame should be 27, got {rep_episode.peak_frame}"
    assert rep_episode.peak_candidate["track_ids"] == [3], f"Representative track should be [3], got {rep_episode.peak_candidate['track_ids']}"


def test_18_tp_anchors_invariant():
    """Assert that 100% (18/18) of primary TP anchor frames remain identical to historical peak frames."""
    baseline_path = os.path.join(PROJECT_ROOT, "artifacts", "post_ab_full_benchmark_sweep.json")
    if not os.path.exists(baseline_path):
        pytest.skip("Historical baseline artifact post_ab_full_benchmark_sweep.json not found")

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tp_results = [r for r in data["per_video_results"] if r.get("is_tp")]
    assert len(tp_results) == 18, f"Expected 18 True Positives in baseline, found {len(tp_results)}"

    invariant_count = 0
    for tp in tp_results:
        cand = tp["matched_candidate"]
        orig_frame = cand["frame"]

        # Recalculate saliency score with temporal decay
        score = calculate_candidate_saliency(cand)
        assert score > 0.0, f"Saliency score should be positive, got {score}"

        # Invariance check: single candidate episode must preserve original peak frame
        episodes = consolidate_candidates_into_episodes(
            raw_candidates=[cand],
            video_id=tp["video_id"],
            gt_frame=tp.get("gt_frame"),
        )
        assert len(episodes) == 1
        assert episodes[0].peak_frame == orig_frame
        invariant_count += 1

    assert invariant_count == 18, f"100% (18/18) invariance required, achieved {invariant_count}/18"


def test_zero_penalty_fallback():
    """Verify that when kinematic onset is unrecorded/unavailable, Delta t falls back to 0.0s."""
    cand_no_onset = {
        "frame": 100,
        "timestamp_s": 3.333,
        "event_type": "collision_candidate",
        "track_ids": [1, 2],
        "confidence": 0.90,
        "signals": {
            "accel_1_ms2": -15.0,
            "bbox_iou": 0.30,
            "ground_distance_m": 1.0,
        },
    }

    score_without_onset = calculate_candidate_saliency(cand_no_onset)

    # Adding explicit onset at candidate frame (dt = 0) should yield identical score
    cand_with_zero_delay = dict(cand_no_onset)
    cand_with_zero_delay["signals"] = dict(cand_no_onset["signals"])
    cand_with_zero_delay["signals"]["decel_onset_frame"] = 100
    score_with_zero_delay = calculate_candidate_saliency(cand_with_zero_delay)

    assert score_without_onset == pytest.approx(score_with_zero_delay, abs=1e-4)
