"""Video Metadata and Stream Probing Module."""

import os
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple
import cv2


@dataclass
class VideoMetadata:
    """Encapsulates video stream parameters and camera identity."""
    video_id: str
    camera_id: str
    resolution: Tuple[int, int]  # (width, height) in pixels
    fps: float
    total_frames: int
    duration_sec: float
    file_path: str

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def aspect_ratio(self) -> float:
        return float(self.width) / float(max(1, self.height))


def probe_video_metadata(
    video_path: str,
    camera_id: Optional[str] = None,
    video_id: Optional[str] = None,
    default_resolution: Tuple[int, int] = (1280, 720),
    default_fps: float = 30.0,
) -> VideoMetadata:
    """Probe video file headers for resolution, frame rate, and duration.
    
    Ensures that downstream calibration and homography engines adapt directly to
    the video's true pixel resolution without hardcoding assumptions.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    # Generate deterministic identifiers if not provided
    stem = os.path.splitext(os.path.basename(video_path))[0]
    vid_id = video_id or stem
    if not camera_id:
        cam_hash = hashlib.sha256(os.path.dirname(video_path).encode("utf-8")).hexdigest()[:8]
        cam_id = f"cam_{cam_hash}"
    else:
        cam_id = camera_id

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return VideoMetadata(
            video_id=vid_id,
            camera_id=cam_id,
            resolution=default_resolution,
            fps=default_fps,
            total_frames=0,
            duration_sec=0.0,
            file_path=os.path.abspath(video_path),
        )

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if w <= 0 or h <= 0:
        w, h = default_resolution
    if fps <= 0.0 or fps > 240.0:
        fps = default_fps
    total_frames = max(0, total_frames)
    duration_sec = float(total_frames / fps) if fps > 0 else 0.0

    return VideoMetadata(
        video_id=vid_id,
        camera_id=cam_id,
        resolution=(w, h),
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        file_path=os.path.abspath(video_path),
    )
