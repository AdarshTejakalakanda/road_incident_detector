"""Vehicle Vocabulary Re-Mapping & Class Confusion Remediation Module.

Iteration K.3: Vehicle Vocabulary Re-Mapping & Class Confusion Remediation
Objective:
Eliminate semantic class confusion (e.g., vehicles misclassified as bottle, chair,
suitcase, or sports ball due to night distortion), group target vehicle classes under
an audited superclass, and use calibrated roadway geometry to recover legitimate objects.

1. Vehicle Superclass Mapping:
   C_vehicle = {"car", "truck", "bus", "motorcycle"}
   Fused superclass confidence: c_vehicle = max_{c in C_vehicle} conf(c)

2. Geometric Roadway Gate:
   - Active Roadway Intersection: bottom-center (x_mid, y_bottom) in Roadway_Polygon
   - BEV Physical Footprint: (W_m, L_m) via PlanarHomography
   - Reclassification Rule:
     If c not in C_vehicle, but:
       - Bottom-center in Roadway_Polygon
       - W_m >= 1.4 m and L_m >= 2.2 m
       - c_vehicle > 0.05
     -> Reclassify as candidate_vehicle_recovered (audit_reclassified = True)
   - Debris Discrimination:
     Objects with ground area < 1.5 m^2 or W_m < 1.4 m or L_m < 2.2 m remain debris / anomalous non-vehicles.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import cv2

from road_surveillance.calibration.homography import PlanarHomography


VEHICLE_SUPERCLASS: Set[str] = {"car", "truck", "bus", "motorcycle"}
HEADLIGHT_CLASSES: Set[str] = {"traffic light", "airplane", "fire hydrant"}
VEHICLE_COCO_IDS: Set[int] = {2, 3, 5, 7}  # COCO class IDs


def compute_bev_footprint(
    box_xyxy: np.ndarray,
    homography: PlanarHomography,
) -> Tuple[float, float, float]:
    """Calculate metric ground footprint (W_m, L_m, Area_m2) from pixel bounding box.

    Args:
        box_xyxy: [x1, y1, x2, y2] in image coordinates.
        homography: Calibrated or dynamic PlanarHomography instance.

    Returns:
        Tuple of (W_m, L_m, Area_m2).
    """
    x1, y1, x2, y2 = float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])
    x_mid = (x1 + x2) / 2.0

    # Lateral ground distance along bottom edge
    p_bl = homography.pixel_to_ground(x1, y2)
    p_br = homography.pixel_to_ground(x2, y2)
    w_m = float(np.hypot(p_br[0] - p_bl[0], p_br[1] - p_bl[1]))

    # Longitudinal ground distance along vertical axis of vehicle
    p_bc = homography.pixel_to_ground(x_mid, y2)
    p_tc = homography.pixel_to_ground(x_mid, y1)
    l_m = float(np.hypot(p_tc[0] - p_bc[0], p_tc[1] - p_bc[1]))

    area_m2 = w_m * l_m
    return round(w_m, 2), round(l_m, 2), round(area_m2, 2)


def is_point_in_polygon(pt: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Test if point (px, py) is strictly inside or on the boundary of polygon."""
    if not polygon or len(polygon) < 3:
        return True  # If no polygon defined, treat as inside active bounds
    poly_np = np.array(polygon, dtype=np.float32)
    res = cv2.pointPolygonTest(poly_np, (float(pt[0]), float(pt[1])), False)
    return bool(res >= 0.0)


class VocabularyGate:
    """Remaps vehicle taxonomy, fuses superclass scores, and filters geometry."""

    VEHICLE_SUPERCLASS = VEHICLE_SUPERCLASS

    def __init__(
        self,
        homography: Optional[PlanarHomography] = None,
        roadway_polygon: Optional[List[Tuple[float, float]]] = None,
        min_width_m: float = 1.4,
        min_length_m: float = 2.2,
        min_area_m2: float = 1.5,
        min_vehicle_logit_conf: float = 0.05,
    ):
        self.homography = homography or PlanarHomography(
            source_points=[[0.0, 720.0], [1280.0, 720.0], [850.0, 300.0], [430.0, 300.0]],
            destination_points=[[-10.0, 0.0], [10.0, 0.0], [10.0, 50.0], [-10.0, 50.0]],
            calib_resolution=(1280, 720),
            target_resolution=(640, 360),
        )
        self.roadway_polygon = roadway_polygon
        self.min_width_m = min_width_m
        self.min_length_m = min_length_m
        self.min_area_m2 = min_area_m2
        self.min_vehicle_logit_conf = min_vehicle_logit_conf

    def get_default_roadway_polygon(self, frame_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Generate default perspective roadway trapezoid based on frame dimensions."""
        h, w = frame_shape[:2]
        # Active highway lane trapezoid
        return [
            (0.0, float(h)),
            (float(w), float(h)),
            (float(w) * 0.75, float(h) * 0.35),
            (float(w) * 0.25, float(h) * 0.35),
        ]

    def audit_and_reclassify_detections(
        self,
        raw_detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        homography: Optional[PlanarHomography] = None,
        roadway_polygon: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """Audit detections, remap superclass confidences, and reclassify road-surface anomalies.

        Args:
            raw_detections: List of detection dicts with keys:
                - "bbox": [x1, y1, x2, y2]
                - "conf": float
                - "class_name": str
                - "class_id": int
            frame_shape: (H, W) or (H, W, C)
            homography: Optional homography override
            roadway_polygon: Optional roadway polygon override

        Returns:
            Audited and reclassified detections list.
        """
        h_engine = homography or self.homography
        if h_engine.target_resolution != (frame_shape[1], frame_shape[0]):
            h_engine = h_engine.adapt_to_resolution((frame_shape[1], frame_shape[0]))

        poly = roadway_polygon or self.roadway_polygon or self.get_default_roadway_polygon(frame_shape)

        # 1. Index all vehicle detections to find secondary vehicle confidences
        vehicle_detections = [
            d for d in raw_detections
            if d.get("class_name", "").lower() in VEHICLE_SUPERCLASS
        ]

        audited_detections = []

        for det in raw_detections:
            bbox = np.array(det["bbox"], dtype=np.float32)
            c_name = det.get("class_name", "").lower()
            conf = float(det.get("conf", 0.0))
            c_id = int(det.get("class_id", -1))

            x1, y1, x2, y2 = bbox
            x_mid = (x1 + x2) / 2.0
            y_bottom = y2

            # Compute metric ground footprint
            w_m, l_m, area_m2 = compute_bev_footprint(bbox, h_engine)

            # Test active roadway intersection
            in_roadway = is_point_in_polygon((x_mid, y_bottom), poly)

            # Compute fused vehicle superclass confidence (c_vehicle)
            if c_name in VEHICLE_SUPERCLASS:
                c_vehicle = conf
                is_vehicle = True
                audit_reclassified = False
                final_class = c_name
                recovery_reason = None
            else:
                # Find secondary vehicle score: check if any vehicle detection overlaps
                c_vehicle = 0.0
                for v_det in vehicle_detections:
                    v_box = np.array(v_det["bbox"], dtype=np.float32)
                    # Compute IoU
                    ix1 = max(x1, v_box[0])
                    iy1 = max(y1, v_box[1])
                    ix2 = min(x2, v_box[2])
                    iy2 = min(y2, v_box[3])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (x2 - x1) * (y2 - y1) + (v_box[2] - v_box[0]) * (v_box[3] - v_box[1]) - inter
                    iou = inter / union if union > 0 else 0.0

                    if iou >= 0.25:
                        c_vehicle = max(c_vehicle, float(v_det["conf"]))

                # Also check if raw detection payload provided secondary logits
                if "secondary_vehicle_conf" in det:
                    c_vehicle = max(c_vehicle, float(det["secondary_vehicle_conf"]))

                raw_cls_norm = c_name.replace(" ", "_")
                contact_point_xy = (x_mid, y_bottom)
                horizon_y_cutoff = min(p[1] for p in poly) if poly else float(frame_shape[0]) * 0.30

                # Patch N.1: Mandatory 3-Condition Geometric Gate
                is_roadway_traffic_light_candidate = (
                    raw_cls_norm == "traffic_light"
                    and in_roadway
                    and (1.2 <= w_m <= 2.6)
                    and (2.2 <= l_m <= 6.5)
                    and (contact_point_xy[1] > horizon_y_cutoff)  # Strictly below road-surface horizon
                )

                # Evaluate Geometric Roadway Gate Reclassification Rule:
                # 1. Bottom-center in Roadway Polygon
                # 2. Footprint satisfies W_m >= 1.4m and L_m >= 2.2m (or headlight/distant vehicle w_m >= 0.8m)
                # 3. c_vehicle > 0.05 or c_name in HEADLIGHT_CLASSES
                # 4. area_m2 >= 1.5m2 (Debris discrimination)
                is_headlight = (c_name in HEADLIGHT_CLASSES)
                meets_dim = (
                    (w_m >= self.min_width_m and l_m >= self.min_length_m)
                    or (is_headlight and w_m >= 0.8 and l_m >= 1.5)
                )
                meets_conf = (c_vehicle >= self.min_vehicle_logit_conf) or is_headlight

                if is_roadway_traffic_light_candidate:
                    is_vehicle = True
                    audit_reclassified = True
                    final_class = "candidate_vehicle_recovered"
                    recovery_reason = "roadway_semantic_remap"
                elif (
                    in_roadway
                    and meets_dim
                    and area_m2 >= self.min_area_m2
                    and meets_conf
                ):
                    is_vehicle = True
                    audit_reclassified = True
                    final_class = "candidate_vehicle_recovered"
                    recovery_reason = "geometric_roadway_gate"
                else:
                    is_vehicle = False
                    audit_reclassified = False
                    final_class = c_name
                    recovery_reason = None

            audited_det = {
                "bbox": [round(float(x), 1) for x in bbox],
                "conf": round(conf, 4),
                "fused_vehicle_conf": round(c_vehicle, 4),
                "class_id": 2 if audit_reclassified else c_id,  # 2 corresponds to car in COCO
                "class_name": final_class,
                "original_class_name": c_name,
                "is_vehicle": is_vehicle,
                "audit_reclassified": audit_reclassified,
                "recovery_reason": recovery_reason,
                "in_roadway": in_roadway,
                "bev_footprint_m": {
                    "width_m": w_m,
                    "length_m": l_m,
                    "area_m2": area_m2,
                },
            }
            audited_detections.append(audited_det)

        return audited_detections
