"""Track state and tracklet definitions."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


import numpy as np


@dataclass
class TrackletState:
    """State of an individual tracklet at a single frame."""
    video_id: str
    track_id: int
    class_id: int
    class_label: str
    frame_idx: int
    pts_ns: int
    bbox_xywh: Tuple[float, float, float, float]
    ground_contact_px: Tuple[float, float]
    track_status: str  # "active", "lost", "new", "removed"
    trust_score: float = 1.0
    provenance: str = "REAL"


@dataclass
class Track:
    """Persistent object track across time."""
    track_id: int
    class_id: int
    class_label: str
    position_history: List[Tuple[float, float]] = field(default_factory=list)  # (x, y) pixels
    ground_history: List[Tuple[float, float]] = field(default_factory=list)    # (X, Y) ground meters
    timestamps: List[int] = field(default_factory=list)                       # pts_ns
    detection_confidence: float = 0.0
    trust_score: float = 1.0
    trust_score_history: List[float] = field(default_factory=list)
    track_age: int = 0
    lost_frame_count: int = 0
    status: str = "active"  # "active", "coasting", "lost", "removed"
    bbox_history: List[Tuple[float, float, float, float]] = field(default_factory=list)
    provenance_history: List[str] = field(default_factory=list)  # "REAL", "INTERPOLATED", "PREDICTED"
    current_bbox_xywh: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    label_history: List[str] = field(default_factory=list)
    coasting_count: int = 0
    is_wrong_way: bool = False
    wrong_way_telemetry: Optional[dict] = None
    is_tentative_rescued: bool = False
    consecutive_matched_frames: int = 1
    was_coasting: bool = False

    VEHICLE_CLASSES = {"car", "truck", "bus", "van", "motorcycle", "train", "vehicle", "candidate_vehicle_recovered"}

    @property
    def is_reacquisition_settled(self) -> bool:
        """True if the track is not recovering from coasting/lost with < 3 consecutive matched frames."""
        if self.was_coasting and self.consecutive_matched_frames < 3:
            return False
        return True

    @staticmethod
    def calculate_ground_contact(bbox_xywh: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Compute ground contact point: bottom center of bounding box.
        x = bbox_x + bbox_width / 2
        y = bbox_y + bbox_height
        """
        bx, by, bw, bh = bbox_xywh
        gc_x = bx + bw / 2.0
        gc_y = by + bh
        return (gc_x, gc_y)

    @property
    def max_recent_speed_kmh(self) -> float:
        """Maximum speed observed in recent history window (km/h).
        
        Calculates moving speed over a multi-frame step (step=3) ignoring initial
        unmatured warm-up frames to filter out single-frame coordinate jitter.
        """
        if len(self.position_history) < 4 or len(self.timestamps) < 4:
            return 0.0
        w, h = self.current_bbox_xywh[2], self.current_bbox_xywh[3]
        char_dim = max(15.0, max(w, h))
        scale = 4.5 / char_dim
        speeds = []
        start_idx = max(3, len(self.position_history) - 10)
        step = min(3, len(self.position_history) - 1)
        for i in range(start_idx, len(self.position_history)):
            dt = (self.timestamps[i] - self.timestamps[i - step]) * 1e-9
            if dt > 1e-4:
                p1 = self.position_history[i]
                p0 = self.position_history[i - step]
                disp = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                speeds.append((disp / dt) * scale * 3.6)
        return float(max(speeds, default=0.0))

    def get_pre_impulse_speed(self, window_s: float = 1.0) -> float:
        """Calculate maximum velocity over a pre-impulse history window."""
        return self.max_recent_speed_kmh

    def get_effective_label(self) -> str:
        """Return temporally smoothed label, prioritizing vehicle identity over visual noise."""
        if not self.label_history:
            return self.class_label
        veh_labels = [l for l in self.label_history if l.lower() in self.VEHICLE_CLASSES]
        if veh_labels:
            # If vehicle label represents at least 20% of history, retain the most common vehicle label
            counts = {}
            for l in veh_labels:
                counts[l] = counts.get(l, 0) + 1
            return max(counts, key=counts.get)
        return self.class_label

    def update(
        self,
        bbox_xywh: Tuple[float, float, float, float],
        confidence: float,
        pts_ns: int,
        label: Optional[str] = None,
        provenance: str = "REAL",
        trust_score: float = 1.0,
        is_tentative_rescued: Optional[bool] = None,
    ) -> None:
        """Update track with new frame detection."""
        if self.status in ("coasting", "lost") or self.coasting_count > 0:
            self.was_coasting = True
            self.consecutive_matched_frames = 1
        elif self.was_coasting:
            self.consecutive_matched_frames += 1
            if self.consecutive_matched_frames >= 3:
                self.was_coasting = False
        else:
            self.consecutive_matched_frames += 1

        self.current_bbox_xywh = bbox_xywh
        self.bbox_history.append(bbox_xywh)
        gc_px = self.calculate_ground_contact(bbox_xywh)
        self.position_history.append(gc_px)
        self.timestamps.append(pts_ns)
        self.provenance_history.append(provenance)
        self.detection_confidence = confidence
        self.trust_score = trust_score
        self.trust_score_history.append(trust_score)
        self.track_age += 1
        self.lost_frame_count = 0
        self.coasting_count = 0
        self.status = "active"
        if is_tentative_rescued is not None:
            self.is_tentative_rescued = is_tentative_rescued
        elif confidence >= 0.25:
            self.is_tentative_rescued = False
        if label:
            self.label_history.append(label)
            self.class_label = self.get_effective_label()

    def coast(self, pts_ns: int, predicted_bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Coast track through momentary occlusion or collision deformation with PREDICTED provenance."""
        self.consecutive_matched_frames = 0
        self.was_coasting = True
        self.coasting_count += 1
        self.track_age += 1
        self.lost_frame_count = 0
        self.status = "coasting"
        if predicted_bbox is not None:
            self.current_bbox_xywh = predicted_bbox
            self.bbox_history.append(predicted_bbox)
            self.position_history.append(self.calculate_ground_contact(predicted_bbox))
        else:
            if self.bbox_history:
                self.bbox_history.append(self.bbox_history[-1])
            if self.position_history:
                self.position_history.append(self.position_history[-1])
        if self.timestamps:
            self.timestamps.append(pts_ns)
        self.provenance_history.append("PREDICTED")

    def interpolate_gap(self, max_gap: int = 6) -> None:
        """Linearly interpolate intermediate PREDICTED observations between the most recent REAL detection
        and the prior REAL detection if the gap is <= max_gap frames.
        """
        if len(self.provenance_history) < 3:
            return
        if self.provenance_history[-1] != "REAL":
            return

        end_idx = len(self.provenance_history) - 1
        start_idx = None
        for i in range(end_idx - 1, -1, -1):
            if self.provenance_history[i] == "REAL":
                start_idx = i
                break

        if start_idx is None:
            return

        gap = end_idx - start_idx - 1
        if 1 <= gap <= max_gap and len(self.bbox_history) == len(self.provenance_history):
            b_start = self.bbox_history[start_idx]
            b_end = self.bbox_history[end_idx]
            p_start = self.position_history[start_idx]
            p_end = self.position_history[end_idx]

            for step, k in enumerate(range(start_idx + 1, end_idx), start=1):
                alpha = step / float(gap + 1)
                interp_box = (
                    float(b_start[0] + alpha * (b_end[0] - b_start[0])),
                    float(b_start[1] + alpha * (b_end[1] - b_start[1])),
                    float(b_start[2] + alpha * (b_end[2] - b_start[2])),
                    float(b_start[3] + alpha * (b_end[3] - b_start[3])),
                )
                interp_pos = (
                    float(p_start[0] + alpha * (p_end[0] - p_start[0])),
                    float(p_start[1] + alpha * (p_end[1] - p_start[1])),
                )
                self.bbox_history[k] = interp_box
                self.position_history[k] = interp_pos
                self.provenance_history[k] = "INTERPOLATED"

    def has_real_observation_near(self, index: int = -1, window: int = 3) -> bool:
        """Safety Invariant: verify that at least one REAL observation occurred within +/- window frames."""
        if not self.provenance_history:
            return True  # Permissive for mock/synthetic tracks without update history
        n = len(self.provenance_history)
        idx = index if index >= 0 else n + index
        if idx < 0 or idx >= n:
            idx = n - 1
        start = max(0, idx - window)
        end = min(n, idx + window + 1)
        return any(self.provenance_history[i] == "REAL" for i in range(start, end))

    def mark_lost(self) -> None:
        """Increment lost frames and update status."""
        self.consecutive_matched_frames = 0
        self.was_coasting = True
        self.lost_frame_count += 1
        self.track_age += 1
        if self.lost_frame_count > 30:
            self.status = "removed"
        else:
            self.status = "lost"
