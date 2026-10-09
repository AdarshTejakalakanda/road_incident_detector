"""End-to-End Pipeline Integration Test."""

import os
import shutil
import pytest
from scripts.create_synthetic_dataset import create_synthetic_incident_video
from scripts.run_pipeline import run_pipeline


def test_end_to_end_pipeline_execution():
    test_video_path = "tests/test_data/synthetic_incident.mp4"
    evidence_dir = "./test_e2e_evidence"
    output_video_dir = "./test_e2e_videos"

    # 1. Generate test video
    create_synthetic_incident_video(output_path=test_video_path, num_frames=30)
    assert os.path.exists(test_video_path)

    # 2. Cleanup previous runs
    if os.path.exists(evidence_dir):
        shutil.rmtree(evidence_dir)
    if os.path.exists(output_video_dir):
        shutil.rmtree(output_video_dir)

    try:
        # 3. Run Pipeline on test video
        run_pipeline(
            config_path="road_surveillance/config/default_config.yaml",
            video_override=test_video_path,
            display=False,
            save_video=True,
            max_videos=1,
            max_frames=25,
        )

        # 4. Verify output video was produced
        assert os.path.exists("./output/videos")
        out_videos = [f for f in os.listdir("./output/videos") if f.endswith(".mp4")]
        assert len(out_videos) >= 1

    finally:
        # Cleanup
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
