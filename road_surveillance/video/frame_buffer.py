"""Rolling Frame Buffer for Evidence Capture."""

from collections import deque
from typing import List, Optional
import numpy as np
from .ingestion import IngestedFrame


class RollingFrameBuffer:
    """Maintains a rolling ring buffer of historical frames for pre-event capture."""

    def __init__(self, max_seconds: float = 10.0, assumed_fps: float = 30.0):
        self.max_seconds = max_seconds
        self.assumed_fps = assumed_fps
        self.max_len = max(30, int(max_seconds * assumed_fps))
        self.buffer: deque[IngestedFrame] = deque(maxlen=self.max_len)

    def append(self, frame: IngestedFrame) -> None:
        """Add ingested frame to rolling buffer, adjusting maxlen if FPS differs."""
        if frame.fps > 0 and frame.fps != self.assumed_fps:
            self.assumed_fps = frame.fps
            new_max = max(30, int(self.max_seconds * self.assumed_fps))
            if new_max != self.max_len:
                self.max_len = new_max
                self.buffer = deque(self.buffer, maxlen=new_max)
        self.buffer.append(frame)

    def get_pre_event_frames(self, duration_sec: float) -> List[IngestedFrame]:
        """Retrieve frames preceding the current moment for given duration in seconds."""
        num_frames = int(duration_sec * self.assumed_fps)
        frames_list = list(self.buffer)
        if len(frames_list) <= num_frames:
            return frames_list
        return frames_list[-num_frames:]

    def clear(self) -> None:
        """Clear the buffer (e.g. between videos)."""
        self.buffer.clear()
