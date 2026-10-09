"""Dataset Video Ingestion Module."""

import os
import hashlib
from dataclasses import dataclass
from typing import Generator, List, Optional
import cv2
import numpy as np


@dataclass
class IngestedFrame:
    camera_id: str
    video_id: str
    frame_idx: int
    pts_ns: int
    fps: float
    image: np.ndarray


class DatasetVideoIngestor:
    """Ingests videos sequentially from local directory or single file."""

    def __init__(
        self,
        path: str,
        recursive: bool = False,
        extensions: Optional[List[str]] = None,
    ):
        self.path = path
        self.recursive = recursive
        self.extensions = [ext.lower() for ext in (extensions or [".mp4", ".avi", ".mov"])]
        self.video_files: List[str] = self._discover_videos()

    def _discover_videos(self) -> List[str]:
        """Discover video files matching extensions."""
        if os.path.isfile(self.path):
            ext = os.path.splitext(self.path)[1].lower()
            if ext in self.extensions:
                return [os.path.abspath(self.path)]
            return []

        discovered = []
        if self.recursive:
            for root, _, files in os.walk(self.path):
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in self.extensions:
                        discovered.append(os.path.abspath(os.path.join(root, f)))
        else:
            if os.path.isdir(self.path):
                for f in sorted(os.listdir(self.path)):
                    full_p = os.path.join(self.path, f)
                    if os.path.isfile(full_p) and os.path.splitext(f)[1].lower() in self.extensions:
                        discovered.append(os.path.abspath(full_p))
        return discovered

    @staticmethod
    def generate_ids(video_path: str) -> tuple[str, str]:
        """Generate deterministic camera_id and video_id from video path."""
        stem = os.path.splitext(os.path.basename(video_path))[0]
        video_id = stem
        # Deterministic camera_id from directory or file hash
        cam_hash = hashlib.sha256(os.path.dirname(video_path).encode("utf-8")).hexdigest()[:8]
        camera_id = f"cam_{cam_hash}"
        return camera_id, video_id

    def stream_video(self, video_path: str) -> Generator[IngestedFrame, None, None]:
        """Stream frames from a single video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0  # Fallback standard FPS if metadata is missing

        camera_id, video_id = self.generate_ids(video_path)
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Precise timestamp calculation
                pts_ns = int(frame_idx * (1e9 / fps))

                yield IngestedFrame(
                    camera_id=camera_id,
                    video_id=video_id,
                    frame_idx=frame_idx,
                    pts_ns=pts_ns,
                    fps=float(fps),
                    image=frame,
                )
                frame_idx += 1
        finally:
            cap.release()

    def stream_all(self) -> Generator[IngestedFrame, None, None]:
        """Sequentially stream frames from all discovered videos."""
        for v_path in self.video_files:
            yield from self.stream_video(v_path)
