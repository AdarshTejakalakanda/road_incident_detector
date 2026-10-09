"""Unit tests for EpisodeEvidenceConsolidator clustering, spatial proximity, and keyframes."""

import os
import shutil
import numpy as np
import pytest

from road_surveillance.evidence.store import EvidenceStore
from road_surveillance.evidence.consolidator import EpisodeEvidenceConsolidator, IncidentEpisode
from road_surveillance.events.base import EventCandidate
from road_surveillance.video.ingestion import IngestedFrame
from road_surveillance.video.frame_buffer import RollingFrameBuffer


@pytest.fixture
def consolidator(tmp_path):
    store = EvidenceStore(root_dir=str(tmp_path / "evidence"))
    buffer = RollingFrameBuffer(max_seconds=5.0)
    return EpisodeEvidenceConsolidator(
        evidence_store=store,
        frame_buffer=buffer,
        max_gap_frames=20,
        post_cooldown_frames=15,
        max_episode_duration_frames=60,
        max_incident_radius_m=15.0,
    )


def _make_frame(frame_idx: int) -> IngestedFrame:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    return IngestedFrame(
        frame_idx=frame_idx,
        pts_ns=int(frame_idx * 66.6e6),
        image=img,
        camera_id="cam_01",
        video_id="vid_01",
        fps=15.0,
    )


def test_consolidator_merges_shared_track(consolidator):
    f1 = _make_frame(10)
    c1 = EventCandidate(
        event_type="collision_candidate",
        video_id="vid_01",
        frame_idx=10,
        track_ids=[5, 8],
        ground_coordinates=(0.0, 10.0),
        signals={"bbox_iou": 0.1, "accel_1_ms2": -5.0},
    )
    ep1 = consolidator.on_event_candidate(c1, f1)
    assert ep1.start_frame == 10
    assert ep1.involved_track_ids == {5, 8}

    # Second trigger 5 frames later sharing track 5
    f2 = _make_frame(15)
    c2 = EventCandidate(
        event_type="single_vehicle_anomaly",
        video_id="vid_01",
        frame_idx=15,
        track_ids=[5],
        ground_coordinates=(1.0, 11.0),
        signals={"anomaly_type": "severe_barrier_impact", "metric_accel_ms2": -15.0},
    )
    ep2 = consolidator.on_event_candidate(c2, f2)
    assert ep2.episode_id == ep1.episode_id
    assert ep2.peak_impact_frame == 15
    assert len(ep2.raw_triggers) == 2


def test_consolidator_rejects_distant_disjoint_tracks(consolidator):
    f1 = _make_frame(10)
    c1 = EventCandidate(
        event_type="collision_candidate",
        video_id="vid_01",
        frame_idx=10,
        track_ids=[1, 2],
        ground_coordinates=(0.0, 40.0),  # Far distance
        signals={"bbox_iou": 0.1},
    )
    ep1 = consolidator.on_event_candidate(c1, f1)

    # Second trigger at same time but 35 meters away with disjoint tracks
    f2 = _make_frame(12)
    c2 = EventCandidate(
        event_type="collision_candidate",
        video_id="vid_01",
        frame_idx=12,
        track_ids=[5, 8],
        ground_coordinates=(0.0, 5.0),  # 35 meters away from (0, 40)
        signals={"bbox_iou": 0.2},
    )
    ep2 = consolidator.on_event_candidate(c2, f2)
    # Must create a separate episode due to spatial separation
    assert ep2.episode_id != ep1.episode_id
    assert len(consolidator.active_episodes) == 2


def test_consolidator_enforces_max_duration_cutoff(consolidator):
    f1 = _make_frame(10)
    c1 = EventCandidate(
        event_type="collision_candidate",
        video_id="vid_01",
        frame_idx=10,
        track_ids=[5, 8],
        ground_coordinates=(0.0, 10.0),
        signals={"bbox_iou": 0.1},
    )
    ep1 = consolidator.on_event_candidate(c1, f1)

    # Advance beyond max_episode_duration_frames (duration limit = 60, frame 80 is 70 frames later)
    f_late = _make_frame(80)
    c_late = EventCandidate(
        event_type="collision_candidate",
        video_id="vid_01",
        frame_idx=80,
        track_ids=[5, 8],
        ground_coordinates=(0.0, 10.0),
        signals={"bbox_iou": 0.1},
    )
    ep_late = consolidator.on_event_candidate(c_late, f_late)
    # Must start a new episode because ep1 reached max duration cutoff
    assert ep_late.episode_id != ep1.episode_id


def test_consolidator_finalization_and_strategic_keyframes(consolidator):
    # Simulate a crash lifecycle: approach at f=20, impact at f=35
    for f_idx in range(10, 55):
        frame = _make_frame(f_idx)
        consolidator.frame_buffer.append(frame)

        if f_idx == 20:
            c = EventCandidate(
                event_type="distant_trajectory_conflict",
                video_id="vid_01",
                frame_idx=20,
                track_ids=[5, 8],
                ground_coordinates=(0.0, 10.0),
                signals={"tca_sec": 1.0},
            )
            consolidator.on_event_candidate(c, frame)
        elif f_idx == 35:
            c = EventCandidate(
                event_type="single_vehicle_anomaly",
                video_id="vid_01",
                frame_idx=35,
                track_ids=[5],
                ground_coordinates=(0.5, 10.2),
                signals={"anomaly_type": "severe_barrier_impact", "metric_accel_ms2": -18.0},
            )
            consolidator.on_event_candidate(c, frame)

        consolidator.process_frame(frame)

    consolidator.flush_all()
    assert len(consolidator.completed_episodes) >= 1
    completed = consolidator.completed_episodes[0]
    assert completed.start_frame == 20
    assert completed.peak_impact_frame == 35
    assert completed.is_finalized is True
