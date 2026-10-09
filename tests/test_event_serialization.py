"""Unit tests for Event and Telemetry JSON Serialization."""

import json
import os
import shutil
import pytest
from road_surveillance.events.base import EventCandidate
from road_surveillance.evidence.store import EvidenceStore


def test_event_metadata_serialization():
    test_dir = "./test_serial_evidence"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    try:
        store = EvidenceStore(root_dir=test_dir)
        candidate = EventCandidate(
            event_id="test-uuid-1234",
            event_type="collision_candidate",
            video_id="video_sample_01",
            frame_idx=142,
            timestamp_ns=4733333333,
            track_ids=[1, 2],
            primary_track_id=1,
            confidence=0.88,
            signals={"decel": -5.2, "tca": 0.4},
            ground_coordinates=(12.5, 34.2),
            lane_id="lane_north",
            is_candidate=True,
            description="Collision test description",
        )

        saved_path = store.save_evidence_package(
            candidate=candidate,
            all_frames=[],
            keyframe=None,
            telemetry_history=[{"frame": 142, "speed": 45.0}],
        )

        meta_path = os.path.join(saved_path, "metadata.json")
        assert os.path.exists(meta_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["event_id"] == "test-uuid-1234"
        assert data["event_type"] == "collision_candidate"
        assert data["video_id"] == "video_sample_01"
        assert data["trigger_frame_idx"] == 142
        assert data["trigger_timestamp_ns"] == 4733333333
        assert data["track_ids"] == [1, 2]
        assert data["confidence"] == 0.88
        assert data["is_candidate"] is True
        assert data["signals"]["decel"] == -5.2
        assert data["ground_coordinates"] == [12.5, 34.2]

        # Verify telemetry file
        telem_path = os.path.join(saved_path, "telemetry", "telemetry_test-uuid-1234.json")
        assert os.path.exists(telem_path)
        with open(telem_path, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        assert len(t_data) == 1
        assert t_data[0]["speed"] == 45.0

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
