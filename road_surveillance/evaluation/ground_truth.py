"""Ground Truth Data Models and Ingestion for Incident Benchmarking."""

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class VideoGroundTruth:
    """Ground truth metadata for an incident video."""
    video_path: str
    video_id: str
    incident_type: str
    rollover: bool
    accident_time_sec: float
    accident_frame: int
    center_x: float  # Normalized [0, 1]
    center_y: float  # Normalized [0, 1]
    bbox_normalized: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    region: str
    scene_layout: str
    weather: str
    day_time: str
    quality: str
    no_frames: int
    duration_sec: float
    height: int
    width: int
    split_in_distribution: str  # "train" or "test"
    split_geo_aware: str        # "train" or "test"


class GroundTruthLoader:
    """Loads and indexes metadata-real.csv for benchmark evaluation."""

    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.records: Dict[str, VideoGroundTruth] = {}
        if os.path.exists(metadata_path):
            self._load()

    def _load(self) -> None:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v_path = row["path"].strip()
                v_id = os.path.splitext(os.path.basename(v_path))[0]

                record = VideoGroundTruth(
                    video_path=v_path,
                    video_id=v_id,
                    incident_type=row.get("type", "unknown").strip(),
                    rollover=bool(int(row.get("rollover", 0))),
                    accident_time_sec=float(row.get("accident_time", 0.0)),
                    accident_frame=int(float(row.get("accident_frame", 0))),
                    center_x=float(row.get("center_x", 0.0)),
                    center_y=float(row.get("center_y", 0.0)),
                    bbox_normalized=(
                        float(row.get("x1", 0.0)),
                        float(row.get("y1", 0.0)),
                        float(row.get("x2", 0.0)),
                        float(row.get("y2", 0.0)),
                    ),
                    region=row.get("region", "").strip(),
                    scene_layout=row.get("scene_layout", "").strip(),
                    weather=row.get("weather", "").strip(),
                    day_time=row.get("day_time", "").strip(),
                    quality=row.get("quality", "").strip(),
                    no_frames=int(float(row.get("no_frames", 0))),
                    duration_sec=float(row.get("duration", 0.0)),
                    height=int(float(row.get("height", 0))),
                    width=int(float(row.get("width", 0))),
                    split_in_distribution=row.get("split_in_distribution", "").strip(),
                    split_geo_aware=row.get("split_geo_aware", "").strip(),
                )
                self.records[v_id] = record

    def get_by_video_id(self, video_id: str) -> Optional[VideoGroundTruth]:
        return self.records.get(video_id)

    def filter_split(self, split: str = "test", split_type: str = "split_in_distribution") -> List[VideoGroundTruth]:
        """Filter records by in-distribution or geo-aware train/test split."""
        results = []
        for rec in self.records.values():
            val = getattr(rec, split_type, None)
            if val == split:
                results.append(rec)
        return results
