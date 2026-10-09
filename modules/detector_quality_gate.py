"""Detector Quality Gate and Unified Telemetry Contract.

Iteration K.4: The Detector Quality Gate & Unified Telemetry Contract
Objective:
Replace raw bounding-box coordinate arrays passed into the tracker with a structured
DetectionRecord contract. Implement dual-threshold admission (high confidence anywhere,
low confidence only if on the roadway with plausible geometry), compute an empirical
trust_score, and carry observation metadata into the tracker state.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from road_surveillance.calibration.homography import PlanarHomography
from modules.vocabulary_gate import (
    VEHICLE_SUPERCLASS,
    VocabularyGate,
    compute_bev_footprint,
    is_point_in_polygon,
)


@dataclass(frozen=True)
class DetectionRecord:
    """Immutable unified detection record contract."""
    detection_id: int
    source_frame: int
    timestamp_s: float
    raw_class: str
    fused_class: str               # "vehicle", "debris", or "unconfirmed"
    raw_confidence: float
    trust_score: float             # Calibrated 0.0 - 1.0 composite
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_area_ratio: float
    is_on_roadway: bool
    ground_dimensions_m: Optional[Tuple[float, float]] = None  # (width_m, length_m)
    audit_reclassified: bool = False       # True if recovered from non-vehicle clutter
    original_class: Optional[str] = None      # Lineage preservation (e.g. "traffic_light")
    recovery_reason: Optional[str] = None     # Root-cause tracking (e.g. "roadway_semantic_remap")
    is_interpolated: bool = False  # Always False from detector; True if filled by tracker
    is_tentative: bool = False     # True if admitted via contextual rescue corridor


class DetectorQualityGate:
    """Tiered dual-threshold admission and trust score calculation engine."""

    def __init__(
        self,
        tau_high: float = 0.25,
        tau_low: float = 0.10,
        w1: float = 0.50,  # Confidence weight
        w2: float = 0.35,  # Roadway prior weight
        w3: float = 0.15,  # Spatial scale weight
        min_width_m: float = 1.2,
        min_length_m: float = 1.8,
        vocab_gate: Optional[VocabularyGate] = None,
    ):
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_width_m = min_width_m
        self.min_length_m = min_length_m
        self.vocab_gate = vocab_gate or VocabularyGate()

    def compute_trust_score(
        self,
        raw_confidence: float,
        is_on_roadway: bool,
        area_ratio: float,
    ) -> float:
        """Formulate empirical trust score T in [0.0, 1.0].

        T = w1 * min(1.0, raw_conf / 0.40) + w2 * I(on_road) + w3 * min(1.0, area_ratio / 0.002)
        """
        c_term = min(1.0, max(0.0, raw_confidence) / 0.40)
        r_term = 1.0 if is_on_roadway else 0.0
        a_term = min(1.0, max(0.0, area_ratio) / 0.002)

        score = (self.w1 * c_term) + (self.w2 * r_term) + (self.w3 * a_term)
        return float(np.clip(score, 0.0, 1.0))

    def process_and_admit(
        self,
        raw_detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp_s: float,
        frame_shape: Tuple[int, int],
        homography: PlanarHomography,
        roadway_polygon: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[List[DetectionRecord], Dict[str, Any]]:
        """Filter, audit, score, and admit detection records via tiered dual-threshold gate.

        Returns:
            Tuple of (admitted_records, quality_gate_telemetry)
        """
        t0 = time.perf_counter()
        h, w = frame_shape[:2]
        frame_area = float(h * w) if (h * w) > 0 else 1.0
        poly = roadway_polygon or self.vocab_gate.get_default_roadway_polygon(frame_shape)

        # 1. First run Vocabulary & Class Remediation Gate
        audited_dets = self.vocab_gate.audit_and_reclassify_detections(
            raw_detections=raw_detections,
            frame_shape=frame_shape,
            homography=homography,
            roadway_polygon=poly,
        )

        admitted_records: List[DetectionRecord] = []
        high_conf_admitted = 0
        low_conf_rescued = 0
        candidate_recovered_admitted = 0
        candidate_recovered_rejected_geometry = 0
        rejected_offroad_low = 0
        rejected_sub_low = 0
        rejected_geometry = 0

        # Collect confirmed vehicles (c >= 0.30) for contextual interaction rescue
        confirmed_veh_boxes = []
        for d in audited_dets:
            if d.get("is_vehicle", False) and float(d.get("conf", 0.0)) >= 0.30:
                cbx1, cby1, cbx2, cby2 = d["bbox"]
                confirmed_veh_boxes.append((cbx1, cby1, cbx2, cby2))

        for idx, det in enumerate(audited_dets):
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            box_area = bw * bh
            area_ratio = box_area / frame_area

            raw_conf = float(det["conf"])
            raw_class = det["original_class_name"]
            is_veh = det["is_vehicle"]
            reclassified = det.get("audit_reclassified", False)

            # Ground metrics
            footprint = det.get("bev_footprint_m", {})
            w_m = footprint.get("width_m", 0.0)
            l_m = footprint.get("length_m", 0.0)
            dims_m = (w_m, l_m) if (w_m > 0 and l_m > 0) else None

            # Test roadway intersection
            x_mid = (x1 + x2) / 2.0
            y_bottom = y2
            on_road = is_point_in_polygon((x_mid, y_bottom), poly)

            # Mandatory 3-Condition Geometric Gate for Roadway Traffic Light Candidate
            contact_point_xy = (x_mid, y_bottom)
            horizon_y_cutoff = min(p[1] for p in poly) if poly else float(h) * 0.30
            raw_cls_norm = raw_class.replace(" ", "_")

            is_roadway_traffic_light_candidate = (
                raw_cls_norm == "traffic_light"
                and on_road
                and (1.2 <= w_m <= 2.6)
                and (2.2 <= l_m <= 6.5)
                and (contact_point_xy[1] > horizon_y_cutoff)  # Strictly below road-surface horizon
            )
            if raw_cls_norm == "traffic_light" and on_road and (contact_point_xy[1] > horizon_y_cutoff):
                if not ((1.2 <= w_m <= 2.6) and (2.2 <= l_m <= 6.5)):
                    candidate_recovered_rejected_geometry += 1

            # Fused class label
            if det.get("class_name") == "candidate_vehicle_recovered" or is_roadway_traffic_light_candidate:
                fused_cls = "candidate_vehicle_recovered"
            elif is_veh or reclassified:
                fused_cls = "vehicle"
            elif raw_class in ("bottle", "cup", "traffic light", "fire hydrant") and not on_road:
                fused_cls = "unconfirmed"
            elif footprint and footprint.get("area_m2", 99.0) < 1.5:
                fused_cls = "debris"
            else:
                fused_cls = "unconfirmed"

            # Compute Trust Score
            trust = self.compute_trust_score(
                raw_confidence=raw_conf,
                is_on_roadway=on_road,
                area_ratio=area_ratio,
            )

            # Tiered Admission Filter:
            admit = False
            orig_cls = raw_class
            recov_reason = None
            is_tentative_rec = False

            if is_roadway_traffic_light_candidate:
                admit = True
                candidate_recovered_admitted += 1
                fused_cls = "candidate_vehicle_recovered"
                reclassified = True
                orig_cls = "traffic_light"
                recov_reason = "roadway_semantic_remap"
            elif det.get("class_name") == "candidate_vehicle_recovered":
                admit = True
                candidate_recovered_admitted += 1
                fused_cls = "candidate_vehicle_recovered"
                reclassified = True
                orig_cls = det.get("original_class_name", raw_class)
                recov_reason = det.get("recovery_reason", "roadway_semantic_remap")
            elif raw_conf >= self.tau_high:
                admit = True
                high_conf_admitted += 1
                orig_cls = raw_class
                recov_reason = None
            elif self.tau_low <= raw_conf < self.tau_high:
                # Rescue only if on active roadway, plausible geometry, and is vehicle/reclassified
                plausible_geom = (w_m >= self.min_width_m and l_m >= self.min_length_m) or (reclassified and w_m >= 0.8 and l_m >= 1.5)
                if on_road and plausible_geom and (is_veh or reclassified):
                    admit = True
                    low_conf_rescued += 1
                    orig_cls = raw_class
                    recov_reason = det.get("recovery_reason")
                else:
                    # Subtask T.3: Interaction-Aware Contextual Rescue Corridor (bJMc3sNP2AU_00)
                    cand_ar = bw / max(1.0, bh)
                    is_plausible_ar = (0.3 <= cand_ar <= 3.5)
                    cand_cx = (x1 + x2) / 2.0
                    cand_cy = (y1 + y2) / 2.0
                    is_in_rescue_corridor = False
                    if is_plausible_ar and (is_veh or reclassified):
                        for (rx1, ry1, rx2, ry2) in confirmed_veh_boxes:
                            if abs(rx1 - x1) < 1e-2 and abs(ry1 - y1) < 1e-2:
                                continue
                            rw = max(1.0, rx2 - rx1)
                            rh = max(1.0, ry2 - ry1)
                            rcx = (rx1 + rx2) / 2.0
                            rcy = (ry1 + ry2) / 2.0
                            dist_px = float(np.hypot(cand_cx - rcx, cand_cy - rcy))
                            max_ref_dim = max(rw, rh)
                            if dist_px <= 1.5 * max_ref_dim:
                                is_in_rescue_corridor = True
                                break

                    if is_in_rescue_corridor:
                        admit = True
                        low_conf_rescued += 1
                        candidate_recovered_admitted += 1
                        fused_cls = "candidate_vehicle_recovered"
                        reclassified = True
                        orig_cls = raw_class
                        recov_reason = "interaction_context_rescue"
                        is_tentative_rec = True
                    elif not on_road:
                        rejected_offroad_low += 1
                    else:
                        rejected_geometry += 1
            else:
                rejected_sub_low += 1

            if admit:
                rec = DetectionRecord(
                    detection_id=idx + 1,
                    source_frame=frame_idx,
                    timestamp_s=round(timestamp_s, 4),
                    raw_class=raw_class,
                    fused_class=fused_cls,
                    raw_confidence=round(raw_conf, 4),
                    trust_score=round(trust, 4),
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    bbox_area_ratio=round(area_ratio, 6),
                    is_on_roadway=on_road,
                    ground_dimensions_m=dims_m,
                    audit_reclassified=reclassified,
                    original_class=orig_cls,
                    recovery_reason=recov_reason,
                    is_interpolated=False,
                    is_tentative=is_tentative_rec,
                )
                admitted_records.append(rec)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            "total_proposals": len(raw_detections),
            "total_admitted": len(admitted_records),
            "high_conf_admitted": high_conf_admitted,
            "low_conf_rescued": low_conf_rescued,
            "candidate_recovered_admitted": candidate_recovered_admitted,
            "candidate_recovered_rejected_geometry": candidate_recovered_rejected_geometry,
            "rejected_offroad_low": rejected_offroad_low,
            "rejected_geometry": rejected_geometry,
            "rejected_sub_low": rejected_sub_low,
            "quality_gate_latency_ms": round(elapsed_ms, 3),
        }

        return admitted_records, telemetry
