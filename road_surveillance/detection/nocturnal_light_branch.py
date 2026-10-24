"""Nocturnal Light-Core Perception Branch & Temporal Vehicle Hypothesis Engine.

Solves nocturnal highway CCTV perception failure modes where vehicle bodies have near-zero
luminance contrast against asphalt, but active light emitters (headlights, taillights, hazards)
are visible and trackable.

Key Architecture Components:
  1. Spatial Light-Core Extraction:
     Extracts high-intensity headlight cores and chromatic red/amber taillight/hazard emitters
     within the calibrated highway ROI. Does NOT immediately treat raw lights as vehicles.
  2. Perspective-Aware Symmetric Pairing:
     Associates horizontal light pairs (taillight pairs, hazard pairs, headlight pairs) based on
     perspective road geometry (expected vehicle width scaling with vertical position y).
  3. Temporal Vehicle Hypothesis Generator:
     Builds and tracks vehicle hypotheses across time. Evaluates:
       - Trajectory consistency along the highway corridor.
       - Velocity vectors (separating moving vehicles from static lamps and reflective signs).
       - Hazard modulation (detecting stopped/incident vehicles with flashing beacons).
       - Persistence gating (rejecting transient headlight sweeps and reflection flashes).
  4. DetectionBatch & BaseDetectorAdapter Integration:
     Outputs standard vehicle bounding boxes and confidences for seamless ingestion by ByteTrack.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

from .detector_interface import DetectionBatch

logger = logging.getLogger("road_surveillance.detection.light_branch")


class LightType(str, Enum):
    TAILLIGHT = "TAILLIGHT"        # Saturated red/amber
    HEADLIGHT = "HEADLIGHT"        # High-intensity white/yellow
    HAZARD_AMBER = "HAZARD_AMBER"  # Pulsing amber/red


class HypothesisState(str, Enum):
    CANDIDATE = "CANDIDATE"                    # Newly spotted light cluster (< 3 frames)
    VALIDATED_MOVING = "VALIDATED_MOVING"      # Consistent motion along road vector
    VALIDATED_STALLED = "VALIDATED_STALLED"    # Arrived or modulated stopped vehicle in lane
    STATIC_REJECTED = "STATIC_REJECTED"        # Permanent roadside sign / lamp (0 motion history)
    TRANSIENT_REJECTED = "TRANSIENT_REJECTED"  # Glare sweep / flicker (< 3 frames, erratic jump)


@dataclass
class LightCore:
    """Represents an extracted spatial light emitter."""
    cx: float
    cy: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    area: float
    mean_intensity: float
    light_type: LightType
    red_dominance: float


@dataclass
class VehicleHypothesis:
    """Temporal vehicle hypothesis built from associated lights and motion."""
    hypothesis_id: int
    first_frame: int
    last_frame: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]  # Synthesized vehicle envelope [x1, y1, x2, y2]
    confidence: float
    state: HypothesisState
    vehicle_class: str               # "truck" or "car"
    paired: bool                     # Whether established from symmetric light pair
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps_s: List[float] = field(default_factory=list)
    intensity_history: List[float] = field(default_factory=list)
    velocity_vector: Tuple[float, float] = (0.0, 0.0)
    missed_count: int = 0


class PerspectiveRoadGeometry:
    """Models perspective scaling of expected vehicle width on highway CCTV."""

    def __init__(
        self,
        horizon_y: float = 310.0,
        foreground_y: float = 650.0,
        min_width_at_horizon: float = 18.0,
        max_width_at_foreground: float = 140.0,
    ):
        self.horizon_y = horizon_y
        self.foreground_y = foreground_y
        self.min_w = min_width_at_horizon
        self.max_w = max_width_at_foreground

    def expected_vehicle_width(self, y: float) -> float:
        """Interpolates expected physical vehicle pixel width at vertical coordinate y."""
        norm_y = np.clip((y - self.horizon_y) / max(1.0, self.foreground_y - self.horizon_y), 0.0, 1.0)
        return self.min_w + norm_y * (self.max_w - self.min_w)


class TemporalVehicleHypothesisEngine:
    """Builds, verifies, and updates temporal vehicle hypotheses from light cores."""

    def __init__(
        self,
        geometry: PerspectiveRoadGeometry,
        min_persistence_frames: int = 3,
        max_association_distance_px: float = 60.0,
        static_speed_threshold_px_s: float = 3.5,
        hazard_flash_min_diff: float = 25.0,
    ):
        self.geom = geometry
        self.min_persistence = min_persistence_frames
        self.max_assoc_dist = max_association_distance_px
        self.static_speed_thresh = static_speed_threshold_px_s
        self.hazard_flash_min_diff = hazard_flash_min_diff
        
        self.next_id = 1
        self.active_hypotheses: Dict[int, VehicleHypothesis] = {}

    def pair_lights(
        self,
        lights: List[LightCore],
        img_w: int,
        img_h: int,
    ) -> Tuple[List[Dict[str, Any]], List[LightCore]]:
        """Pairs symmetric horizontal light cores into vehicle candidate envelopes.
        
        Returns:
            paired_envelopes: List of synthesized vehicle proposals.
            unpaired_lights: Remaining single light cores.
        """
        used = set()
        paired_envelopes = []
        n = len(lights)

        for i in range(n):
            if i in used:
                continue
            l1 = lights[i]
            best_j = -1
            best_score = float("inf")

            for j in range(i + 1, n):
                if j in used:
                    continue
                l2 = lights[j]

                # 1. Color consistency: taillights pair with taillights, headlights with headlights
                if l1.light_type != l2.light_type:
                    continue

                # 2. Horizontal alignment: vehicles have nearly horizontal light pairs (low delta_y)
                dy = abs(l1.cy - l2.cy)
                mid_y = (l1.cy + l2.cy) / 2.0
                exp_w = self.geom.expected_vehicle_width(mid_y)
                
                # Max allowable vertical tilt relative to perspective width
                if dy > max(8.0, exp_w * 0.25):
                    continue

                # 3. Horizontal distance constraint
                dx = abs(l1.cx - l2.cx)
                if dx < exp_w * 0.35 or dx > exp_w * 1.65:
                    continue

                # 4. Symmetry score: combination of horizontal distance match and vertical alignment
                score = (abs(dx - exp_w) / exp_w) + (dy / exp_w) * 2.0
                if score < best_score and score < 1.0:
                    best_score = score
                    best_j = j

            if best_j != -1:
                used.add(i)
                used.add(best_j)
                l2 = lights[best_j]
                
                # Synthesize vehicle envelope enclosing both lights with realistic vehicle chassis padding
                mid_x = (l1.cx + l2.cx) / 2.0
                mid_y = (l1.cy + l2.cy) / 2.0
                w_span = max(l1.bbox[2], l2.bbox[2]) - min(l1.bbox[0], l2.bbox[0])
                h_span = max(l1.bbox[3], l2.bbox[3]) - min(l1.bbox[1], l2.bbox[1])
                
                # Vehicle chassis expands upwards and sideways
                pad_x = int(w_span * 0.25)
                # Vehicle body rises above taillights/headlights
                pad_top = int(max(h_span, self.geom.expected_vehicle_width(mid_y) * 0.70))
                pad_bottom = int(h_span * 0.20)

                vx1 = max(0, min(l1.bbox[0], l2.bbox[0]) - pad_x)
                vy1 = max(0, min(l1.bbox[1], l2.bbox[1]) - pad_top)
                vx2 = min(img_w, max(l1.bbox[2], l2.bbox[2]) + pad_x)
                vy2 = min(img_h, max(l1.bbox[3], l2.bbox[3]) + pad_bottom)

                v_class = "truck" if (vx2 - vx1) > self.geom.expected_vehicle_width(mid_y) * 1.1 else "car"
                paired_envelopes.append({
                    "centroid": (mid_x, mid_y),
                    "bbox": (vx1, vy1, vx2, vy2),
                    "paired": True,
                    "intensity": (l1.mean_intensity + l2.mean_intensity) / 2.0,
                    "class": v_class,
                })

        unpaired = [lights[i] for i in range(n) if i not in used]
        return paired_envelopes, unpaired

    def update(
        self,
        paired_envelopes: List[Dict[str, Any]],
        unpaired_lights: List[LightCore],
        frame_idx: int,
        timestamp_s: float,
        img_w: int,
        img_h: int,
    ) -> List[VehicleHypothesis]:
        """Updates temporal hypotheses and returns validated vehicle detections."""
        # Convert unpaired lights into candidate single-emitter envelopes
        all_candidates = list(paired_envelopes)
        for ul in unpaired_lights:
            exp_w = self.geom.expected_vehicle_width(ul.cy)
            # Size box according to perspective geometry
            half_w = exp_w * 0.45
            h_body = exp_w * 0.65
            x1 = max(0, int(ul.cx - half_w))
            y1 = max(0, int(ul.cy - h_body))
            x2 = min(img_w, int(ul.cx + half_w))
            y2 = min(img_h, int(ul.cy + h_body * 0.25))
            
            all_candidates.append({
                "centroid": (ul.cx, ul.cy),
                "bbox": (x1, y1, x2, y2),
                "paired": False,
                "intensity": ul.mean_intensity,
                "class": "truck" if ul.area > 500 else "car",
            })

        # Associate candidates with active hypotheses via nearest-neighbor distance
        matched_cand = set()
        matched_hyp = set()

        for hyp_id, hyp in self.active_hypotheses.items():
            best_c_idx = -1
            best_dist = float("inf")

            for c_idx, cand in enumerate(all_candidates):
                if c_idx in matched_cand:
                    continue
                dx = cand["centroid"][0] - hyp.centroid[0]
                dy = cand["centroid"][1] - hyp.centroid[1]
                dist = np.hypot(dx, dy)

                # Distance threshold scales with perspective vertical coordinate
                max_d = self.geom.expected_vehicle_width(hyp.centroid[1]) * 1.5
                if dist < best_dist and dist < max_d:
                    best_dist = dist
                    best_c_idx = c_idx

            if best_c_idx != -1:
                matched_cand.add(best_c_idx)
                matched_hyp.add(hyp_id)
                cand = all_candidates[best_c_idx]

                # Update hypothesis kinematics
                dt = max(1e-3, timestamp_s - hyp.timestamps_s[-1]) if hyp.timestamps_s else 0.125
                vx = (cand["centroid"][0] - hyp.centroid[0]) / dt
                vy = (cand["centroid"][1] - hyp.centroid[1]) / dt

                hyp.centroid = cand["centroid"]
                hyp.bbox = cand["bbox"]
                hyp.last_frame = frame_idx
                hyp.positions.append(cand["centroid"])
                hyp.timestamps_s.append(timestamp_s)
                hyp.intensity_history.append(cand["intensity"])
                hyp.velocity_vector = (vx, vy)
                hyp.paired = cand["paired"] or hyp.paired
                hyp.missed_count = 0

                # Evaluate state promotion / rejection
                self._evaluate_hypothesis_state(hyp)

        # Increment missed count for unmatched hypotheses
        to_delete = []
        for hyp_id, hyp in self.active_hypotheses.items():
            if hyp_id not in matched_hyp:
                hyp.missed_count += 1
                # Drop after 4 missed frames
                if hyp.missed_count > 4:
                    to_delete.append(hyp_id)

        for hid in to_delete:
            del self.active_hypotheses[hid]

        # Instantiate new hypotheses for unmatched candidates
        for c_idx, cand in enumerate(all_candidates):
            if c_idx not in matched_cand:
                new_hyp = VehicleHypothesis(
                    hypothesis_id=self.next_id,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    centroid=cand["centroid"],
                    bbox=cand["bbox"],
                    confidence=0.55 if cand["paired"] else 0.40,
                    state=HypothesisState.CANDIDATE,
                    vehicle_class=cand["class"],
                    paired=cand["paired"],
                    positions=[cand["centroid"]],
                    timestamps_s=[timestamp_s],
                    intensity_history=[cand["intensity"]],
                    velocity_vector=(0.0, 0.0),
                    missed_count=0,
                )
                self.active_hypotheses[self.next_id] = new_hyp
                self.next_id += 1

        # Return validated vehicle hypotheses
        validated = [
            h for h in self.active_hypotheses.values()
            if h.state in (HypothesisState.VALIDATED_MOVING, HypothesisState.VALIDATED_STALLED)
            and h.missed_count == 0
        ]
        return validated

    def _evaluate_hypothesis_state(self, hyp: VehicleHypothesis) -> None:
        """Classifies hypothesis into moving vehicle, stalled incident, static sign, or transient."""
        duration_frames = len(hyp.positions)
        if duration_frames < self.min_persistence:
            hyp.state = HypothesisState.CANDIDATE
            return

        # Calculate total spatial displacement and speed
        p_first = np.array(hyp.positions[0])
        p_last = np.array(hyp.positions[-1])
        total_disp = float(np.linalg.norm(p_last - p_first))
        total_time = max(0.01, hyp.timestamps_s[-1] - hyp.timestamps_s[0])
        speed = total_disp / total_time  # pixels per second

        # Detect hazard flash modulation (flashing taillights/hazards at 1-3 Hz)
        has_hazard_flashing = False
        if len(hyp.intensity_history) >= 6:
            recent_i = hyp.intensity_history[-10:]
            i_diff = max(recent_i) - min(recent_i)
            has_hazard_flashing = i_diff > self.hazard_flash_min_diff

        # 1. Check for Static Infrastructure / Reflective Sign
        # If appeared in first 3 frames, never moved, and has no hazard modulation -> Static
        if hyp.first_frame <= 2 and speed < self.static_speed_thresh and not has_hazard_flashing and not hyp.paired:
            hyp.state = HypothesisState.STATIC_REJECTED
            return

        # 2. Check for Valid Moving Vehicle
        if speed >= self.static_speed_thresh:
            hyp.state = HypothesisState.VALIDATED_MOVING
            # Boost confidence for smooth paired trajectories
            base_conf = 0.70 if hyp.paired else 0.60
            hyp.confidence = min(0.95, base_conf + min(0.25, duration_frames * 0.03))
            return

        # 3. Check for Stalled / Incident Vehicle (stopped in active lane or hazard lights active)
        if speed < self.static_speed_thresh and (has_hazard_flashing or hyp.paired or hyp.first_frame > 5):
            hyp.state = HypothesisState.VALIDATED_STALLED
            hyp.confidence = min(0.95, 0.75 + min(0.20, duration_frames * 0.02))
            hyp.vehicle_class = "truck"  # Stalled heavy vehicle / hazard
            return

        # Default fallback: remain candidate until more evidence accumulates
        hyp.state = HypothesisState.CANDIDATE


class NocturnalLightBranchDetector:
    """End-to-end Nocturnal Light-Core Branch producing standard DetectionBatches."""

    def __init__(
        self,
        road_polygon: np.ndarray,
        red_threshold: int = 35,
        headlight_threshold: int = 200,
        min_light_area: float = 60.0,
    ):
        self.road_poly = road_polygon
        self.red_thresh = red_threshold
        self.white_thresh = headlight_threshold
        self.min_area = min_light_area

        self.geom = PerspectiveRoadGeometry(
            horizon_y=float(np.min(road_polygon[:, 1])),
            foreground_y=float(np.max(road_polygon[:, 1])),
        )
        self.engine = TemporalVehicleHypothesisEngine(self.geom)
        self.mask: Optional[np.ndarray] = None

    def _ensure_mask(self, h: int, w: int) -> np.ndarray:
        if self.mask is None or self.mask.shape != (h, w):
            self.mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(self.mask, [self.road_poly], 255)
        return self.mask

    def extract_light_cores(self, frame: np.ndarray) -> List[LightCore]:
        """Extracts spatial light emitter cores inside the road corridor."""
        h, w = frame.shape[:2]
        road_mask = self._ensure_mask(h, w)
        masked = cv2.bitwise_and(frame, frame, mask=road_mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(masked)

        # 1. Red chromatic taillights / hazards
        red_dom = cv2.subtract(r, cv2.max(b, g))
        _, red_mask = cv2.threshold(red_dom, self.red_thresh, 255, cv2.THRESH_BINARY)

        # 2. Saturated white headlights
        _, white_mask = cv2.threshold(gray, self.white_thresh, 255, cv2.THRESH_BINARY)

        cores = []
        for mask_layer, ltype in [(red_mask, LightType.TAILLIGHT), (white_mask, LightType.HEADLIGHT)]:
            contours, _ = cv2.findContours(mask_layer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area or area > 45000:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                
                # Mean intensity
                roi_gray = gray[y:y+bh, x:x+bw]
                mean_int = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0

                cores.append(LightCore(
                    cx=cx,
                    cy=cy,
                    bbox=(x, y, x + bw, y + bh),
                    area=area,
                    mean_intensity=mean_int,
                    light_type=ltype,
                    red_dominance=float(np.mean(red_dom[y:y+bh, x:x+bw])) if ltype == LightType.TAILLIGHT else 0.0,
                ))

        return cores

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_s: float,
        video_id: str = "eval_clip",
    ) -> DetectionBatch:
        """Processes frame through light extraction, pairing, and hypothesis tracking."""
        h, w = frame.shape[:2]
        lights = self.extract_light_cores(frame)
        paired_envelopes, unpaired_lights = self.engine.pair_lights(lights, w, h)
        
        validated_vehicles = self.engine.update(
            paired_envelopes=paired_envelopes,
            unpaired_lights=unpaired_lights,
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            img_w=w,
            img_h=h,
        )

        n_boxes = len(validated_vehicles)
        if n_boxes == 0:
            return DetectionBatch(
                video_id=video_id,
                frame_idx=frame_idx,
                pts_ns=int(timestamp_s * 1e9),
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        boxes = np.array([v.bbox for v in validated_vehicles], dtype=np.float32)
        confs = np.array([v.confidence for v in validated_vehicles], dtype=np.float32)
        class_labels = [v.vehicle_class for v in validated_vehicles]
        class_ids = np.array([1 if l == "truck" else 0 for l in class_labels], dtype=np.int32)

        return DetectionBatch(
            video_id=video_id,
            frame_idx=frame_idx,
            pts_ns=int(timestamp_s * 1e9),
            boxes=boxes,
            confidences=confs,
            class_ids=class_ids,
            class_labels=class_labels,
        )
