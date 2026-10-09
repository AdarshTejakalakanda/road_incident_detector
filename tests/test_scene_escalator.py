import pytest
from modules.scene_escalator import SceneEscalator, CandidateProposal, TrackState


def test_lowest_tp_retention():
    escalator = SceneEscalator(tau_aftermath=0.50)
    cand = CandidateProposal(
        frame_idx=133,
        event_type='collision_candidate',
        track_ids=[1, 10],
        video_id='iA_u73BG4ZM_00',
        signals={
            'canonical_type': 'Type 2',
            'v_entry': 24.5,
            'dv_sust': 24.0,
            'tau_rest': 1.00,
            'delta_theta': 18.5,
            'tau_overlap': 0.65,
            'delta_d_lat': 0.43,
            'push_disp': 1.10,
            'upstream_react': 3,
        }
    )
    result = escalator.evaluate_candidate(cand, track_history={})
    assert result.score >= 0.80
    assert result.escalated is True
    assert result.canonical_type == 'Type 2'
    assert result.channels['phi_dv'] == 1.0
    assert result.channels['phi_rest'] == 1.0


def test_highest_fp_suppression():
    escalator = SceneEscalator(tau_aftermath=0.50)
    cand = CandidateProposal(
        frame_idx=275,
        event_type='collision_candidate',
        track_ids=[77, 96],
        video_id='5uh8-3q51uI_00',
        signals={
            'canonical_type': 'Type 3',
            'v_entry': 24.7,
            'dv_sust': 7.4,
            'tau_rest': 0.00,
            'delta_theta': 12.0,
            'tau_overlap': 0.11,
            'delta_d_lat': 0.45,
            'push_disp': 0.00,
            'upstream_react': 0,
        }
    )
    result = escalator.evaluate_candidate(cand, track_history={})
    assert result.score <= 0.25
    assert result.escalated is False
    assert result.channels['phi_rest'] == 0.0


def test_high_speed_near_miss_safeguard():
    escalator = SceneEscalator(tau_aftermath=0.50)
    cand = CandidateProposal(
        frame_idx=625,
        event_type='collision_candidate',
        track_ids=[169, 188],
        video_id='5uh8-3q51uI_00',
        signals={
            'canonical_type': 'Type 1',
            'v_entry': 74.3,
            'dv_sust': 37.0,
            'tau_rest': 0.00,
            'delta_theta': 8.5,
            'tau_overlap': 0.00,
            'delta_d_lat': 0.45,
            'push_disp': 0.00,
            'upstream_react': 0,
        }
    )
    result = escalator.evaluate_candidate(cand, track_history={})
    assert result.score <= 0.35
    assert result.escalated is False
    assert result.safeguards['dv_denominator'] > 40.0
    assert result.channels['phi_dv'] < 0.90


def test_crawl_queue_safeguard():
    escalator = SceneEscalator(tau_aftermath=0.50)
    cand = CandidateProposal(
        frame_idx=609,
        event_type='collision_candidate',
        track_ids=[159, 195],
        video_id='5uh8-3q51uI_00',
        signals={
            'canonical_type': 'Type 3',
            'v_entry': 5.0,
            'dv_sust': 0.5,
            'tau_rest': 1.00,
            'delta_theta': 4.0,
            'tau_overlap': 0.10,
            'delta_d_lat': 0.20,
            'push_disp': 0.00,
            'upstream_react': 0,
        }
    )
    result = escalator.evaluate_candidate(cand, track_history={})
    assert result.channels['phi_rest'] == 0.0
    assert result.safeguards['crawl_guard_active'] is True
    assert result.safeguards['crawl_override_passed'] is False
    assert result.score <= 0.20
    assert result.escalated is False


def test_track_history_continuous_evaluation():
    escalator = SceneEscalator(tau_aftermath=0.50)
    cand = CandidateProposal(
        frame_idx=100,
        event_type='collision_candidate',
        track_ids=[1, 2],
        video_id='synthetic_tbone',
        signals={'is_tbone': True}
    )

    history = {1: [], 2: []}
    for f in range(100, 146):
        spd1 = max(0.0, 40.0 - (f - 100) * 4.0)
        hdg1 = 90.0 + min(70.0, (f - 100) * 7.0)
        history[1].append(TrackState(frame_idx=f, pts_ns=int(f * 3.33e7), bbox=(100, 100, 40, 20), ground_pos=(10.0 + f * 0.05, 5.0), speed_kmh=spd1, heading_deg=hdg1))

        spd2 = max(0.0, 30.0 - (f - 100) * 3.0)
        hdg2 = 180.0
        history[2].append(TrackState(frame_idx=f, pts_ns=int(f * 3.33e7), bbox=(105, 102, 40, 20), ground_pos=(10.2 + f * 0.08, 5.1), speed_kmh=spd2, heading_deg=hdg2))

    result = escalator.evaluate_candidate(cand, history, fps=30.0)
    assert result.canonical_type == 'Type 3'
    assert result.score >= 0.70
    assert result.escalated is True
    assert result.telemetry['tau_rest'] > 0.50
    assert result.telemetry['delta_theta'] >= 60.0
