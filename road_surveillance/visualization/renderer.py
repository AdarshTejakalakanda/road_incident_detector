"""Playback and Video Export Renderer."""

import os
import logging
from typing import Optional
import cv2
import numpy as np

logger = logging.getLogger("road_surveillance.visualization.renderer")


class PlaybackRenderer:
    """Handles real-time display and export of processed annotated video files."""

    def __init__(
        self,
        display: bool = False,
        save_processed_video: bool = True,
        realtime: bool = False,
        output_dir: str = "./output/videos",
    ):
        self.display = display
        self.save_processed_video = save_processed_video
        self.realtime = realtime
        self.output_dir = os.path.abspath(output_dir)

        if self.save_processed_video:
            os.makedirs(self.output_dir, exist_ok=True)

        self.writer: Optional[cv2.VideoWriter] = None
        self.current_video_id: Optional[str] = None
        self.window_name = "Road Surveillance Incident Detection"

    def setup_video_writer(self, video_id: str, fps: float, width: int, height: int) -> None:
        """Initialize video writer for a new video stream."""
        if not self.save_processed_video:
            return

        if self.writer is not None:
            self.writer.release()

        self.current_video_id = video_id
        out_path = os.path.join(self.output_dir, f"{video_id}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = fps if fps > 0 else 30.0
        self.writer = cv2.VideoWriter(out_path, fourcc, float(effective_fps), (width, height))
        logger.info(f"Initialized output video writer: {out_path} ({width}x{height} @ {effective_fps:.1f} FPS)")

    def render(self, frame: np.ndarray, fps: float) -> bool:
        """Render frame to window and/or disk. Returns False if user requested quit."""
        # Write to MP4 file
        if self.writer is not None:
            self.writer.write(frame)

        # Display window
        if self.display:
            cv2.imshow(self.window_name, frame)
            delay = int(1000.0 / fps) if (self.realtime and fps > 0) else 1
            key = cv2.waitKey(max(1, delay)) & 0xFF
            if key == ord("q"):
                logger.info("User requested quit from display window.")
                return False
            elif key == ord(" "):  # Pause / step
                cv2.waitKey(0)

        return True

    def close(self) -> None:
        """Release writers and destroy windows."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
