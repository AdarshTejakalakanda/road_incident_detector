"""Visualization Annotator Module."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from ..tracking.track_state import Track, TrackletState
from ..geometry.spatial_metrics import TrackKinematics
from ..calibration.lane_manager import LaneManager, Lane
from ..calibration.homography import PlanarHomography
from ..events.base import EventCandidate


class FrameAnnotator:
    """Renders visual overlays for bounding boxes, trajectories, lanes, HUD, and alerts."""

    COLOR_PALETTE = [
        (255, 105, 65),   # Royal blue
        (50, 205, 50),    # Lime green
        (0, 165, 255),    # Orange
        (238, 130, 238),  # Violet
        (0, 215, 255),    # Gold
        (255, 20, 147),   # Deep pink
        (0, 255, 255),    # Yellow
    ]

    def __init__(
        self,
        homography: Optional[PlanarHomography] = None,
        lane_manager: Optional[LaneManager] = None,
    ):
        self.homography = homography
        self.lane_manager = lane_manager

    def annotate(
        self,
        image: np.ndarray,
        tracklets: List[TrackletState],
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        active_candidates: List[EventCandidate],
        fps: float,
        frame_idx: int,
        pts_ns: int,
    ) -> np.ndarray:
        """Render complete annotation layer on frame."""
        canvas = image.copy()

        # 1. Draw lane polygons and permitted directions
        if self.lane_manager:
            for lane in self.lane_manager.lanes:
                self._draw_lane(canvas, lane)

        # 2. Draw active tracklets, ground contact points, and trajectory trails
        for tracklet in tracklets:
            tid = tracklet.track_id
            track = tracks.get(tid)
            kin = kinematics.get(tid)
            color = self.COLOR_PALETTE[tid % len(self.COLOR_PALETTE)]

            # Bounding box
            bx, by, bw, bh = tracklet.bbox_xywh
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Ground contact point
            gc_x, gc_y = int(tracklet.ground_contact_px[0]), int(tracklet.ground_contact_px[1])
            cv2.circle(canvas, (gc_x, gc_y), 4, (0, 0, 255), -1)
            cv2.circle(canvas, (gc_x, gc_y), 6, color, 1)

            # Trajectory history trail
            if track and len(track.position_history) >= 2:
                pts = np.array(track.position_history[-25:], dtype=np.int32)
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

            # Telemetry label (ID, class, speed km/h)
            speed_str = f"{kin.speed_kmh:.1f} km/h" if kin else ""
            label_text = f"ID:{tid} {tracklet.class_label} {speed_str}".strip()

            # Background label tag
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw diagnostic HUD (top-left)
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (10, 10), (320, 85), (0, 0, 0), -1)
        cv2.addWeighted(hud_bg, 0.6, canvas, 0.4, 0, canvas)

        time_sec = pts_ns * 1e-9
        cv2.putText(canvas, f"Frame: {frame_idx} | FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(canvas, f"Time: {time_sec:.2f}s", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"Tracks: {len(tracklets)} active", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # 4. Draw prominent Event Candidate Alert Banner (top center/right)
        if active_candidates:
            self._draw_alert_banner(canvas, active_candidates)

        return canvas

    def _draw_lane(self, canvas: np.ndarray, lane: Lane) -> None:
        """Render lane polygon and direction arrow."""
        if not lane.polygon_pixel and lane.polygon_ground and self.homography:
            pts_g = np.array(lane.polygon_ground, dtype=np.float64)
            lane.polygon_pixel = [tuple(p) for p in self.homography.batch_ground_to_pixel(pts_g)]

        if lane.polygon_pixel and len(lane.polygon_pixel) >= 3:
            pts = np.array(lane.polygon_pixel, dtype=np.int32)
            color = (0, 255, 128) if lane.is_active else (128, 128, 128)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)

            # Draw lane ID
            centroid_x = int(np.mean(pts[:, 0]))
            centroid_y = int(np.mean(pts[:, 1]))
            cv2.putText(canvas, lane.lane_id, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _draw_alert_banner(self, canvas: np.ndarray, candidates: List[EventCandidate]) -> None:
        """Draw high-visibility warning banner when candidates are active."""
        h, w = canvas.shape[:2]
        banner_h = 36 * len(candidates)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h + 10), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        for i, cand in enumerate(candidates):
            y = 26 + i * 32
            alert_text = f"EVENT CANDIDATE [{cand.event_type.upper()}]: {cand.description}"
            cv2.putText(canvas, alert_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
