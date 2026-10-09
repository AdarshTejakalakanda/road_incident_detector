"""Iteration N - Patch N.2: Motion-Guided Temporal Differencing ROI Crop Gate.

Objective:
Recover distant, low-contrast micro-vehicles (< 1024 px^2, width ~30 px) that fail
standard detector strides on surveillance video (e.g. nAXTthLfgtI_00) without
running global high-resolution inference on full frames.

Operational Invariants:
1. Global detector resolution remains frozen at 640p.
2. Global confidence thresholds remain frozen (tau_low = 0.10).
3. Zoomed ROI inference is triggered ONLY when persistent, clustered motion is
   detected inside the active roadway mask where the primary detector produced
   zero overlapping vehicle bounding boxes.
4. Compute Budget: Throughput remains >= 16 FPS on CPU by evaluating a maximum of
   1 ROI crop per frame.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from ultralytics import YOLO


def extract_roadway_motion_clusters(
    frame_curr: np.ndarray,
    frame_prev: np.ndarray,
    roadway_mask: np.ndarray,
    min_cluster_area: int = 250,
    max_cluster_area: int = 3500,
) -> List[Tuple[int, int, int, int, float]]:
    """Extract motion regions of interest from frame difference on the active roadway.

    Args:
        frame_curr: Current BGR frame.
        frame_prev: Previous BGR frame.
        roadway_mask: Binary mask (uint8) where 255 represents active roadway travel lanes.
        min_cluster_area: Minimum contour area threshold for motion clusters.
        max_cluster_area: Maximum contour area threshold to suppress whole-scene lighting shifts.

    Returns:
        List of tuples (x1, y1, x2, y2, contour_area) with 50% spatial padding.
    """
    if frame_curr is None or frame_prev is None or roadway_mask is None:
        return []

    # 1. Grayscale difference on masked roadway
    diff = cv2.absdiff(frame_curr, frame_prev)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=roadway_mask)

    # 2. Threshold motion energy (adapt to local noise)
    _, thresh = cv2.threshold(masked_diff, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 3. Find connected motion components
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_rois: List[Tuple[int, int, int, int, float]] = []

    h, w = frame_curr.shape[:2]

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if min_cluster_area <= area <= max_cluster_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Expand bounding box with 50% spatial padding for context
            pad_x = int(bw * 0.5)
            pad_y = int(bh * 0.5)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            motion_rois.append((x1, y1, x2, y2, area))

    return motion_rois


def compute_box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute standard 2D Intersection over Union between two [x1, y1, x2, y2] boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 1e-6 else 0.0


def is_box_overlapping_cluster(cluster_box: Tuple[int, int, int, int], det_box: Tuple[float, float, float, float]) -> bool:
    """Check if a detection box overlaps or has its center inside a cluster box."""
    cx = (det_box[0] + det_box[2]) / 2.0
    cy = (det_box[1] + det_box[3]) / 2.0
    if cluster_box[0] <= cx <= cluster_box[2] and cluster_box[1] <= cy <= cluster_box[3]:
        return True
    return compute_box_iou(cluster_box, det_box) > 0.05


class MotionROIGate:
    """Motion-Guided Temporal Differencing and Localized Re-Inference Engine."""

    TARGET_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van", "train"}

    def __init__(
        self,
        min_cluster_area: int = 250,
        max_cluster_area: int = 3500,
        persistence_threshold: int = 4,
        crop_resolution: int = 320,
        crop_conf_threshold: float = 0.05,
        max_crops_per_frame: int = 1,
        centroid_match_dist: float = 80.0,
        min_crop_interval: int = 1,
    ):
        self.min_cluster_area = min_cluster_area
        self.max_cluster_area = max_cluster_area
        self.persistence_threshold = persistence_threshold
        self.crop_resolution = crop_resolution
        self.crop_conf_threshold = crop_conf_threshold
        self.max_crops_per_frame = max_crops_per_frame
        self.centroid_match_dist = centroid_match_dist
        self.min_crop_interval = min_crop_interval

        self._tracked_clusters: List[Dict[str, Any]] = []
        self._next_cluster_id: int = 1
        self._prev_frame: Optional[np.ndarray] = None
        self._roadway_mask: Optional[np.ndarray] = None
        self._last_crop_frame: int = -999

    def reset(self) -> None:
        """Reset internal state between video streams."""
        self._tracked_clusters.clear()
        self._next_cluster_id = 1
        self._prev_frame = None
        self._roadway_mask = None
        self._last_crop_frame = -999

    def _ensure_roadway_mask(self, frame_shape: Tuple[int, int], roadway_polygon: Optional[List[Tuple[float, float]]]) -> np.ndarray:
        h, w = frame_shape[:2]
        if self._roadway_mask is not None and self._roadway_mask.shape == (h, w):
            return self._roadway_mask

        mask = np.zeros((h, w), dtype=np.uint8)
        poly = roadway_polygon
        if not poly:
            from modules.vocabulary_gate import VocabularyGate
            poly = VocabularyGate().get_default_roadway_polygon(frame_shape)
        if poly:
            poly_np = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [poly_np], 255)
        self._roadway_mask = mask
        return mask

    def process_and_recover(
        self,
        frame_curr: np.ndarray,
        frame_idx: int,
        primary_raw_detections: List[Dict[str, Any]],
        roadway_polygon: List[Tuple[float, float]],
        yolo_model: YOLO,
        active_track_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Evaluate motion clusters and run localized zoomed re-inference if triggered.

        Args:
            frame_curr: Native resolution BGR frame.
            frame_idx: Frame index in the stream.
            primary_raw_detections: List of detection dicts from primary 640p pass.
            roadway_polygon: Polygon defining the active roadway surface.
            yolo_model: Ultralytics YOLO model instance.
            active_track_boxes: Optional list of bounding boxes from currently active tracks.

        Returns:
            Tuple of:
                - recovered_detections: List of recovered Detection dicts (if any)
                - telemetry: Audit metadata regarding cluster count, evaluations, hits
        """
        recovered_detections: List[Dict[str, Any]] = []
        telemetry = {
            "motion_clusters_extracted": 0,
            "persistent_clusters_tracked": 0,
            "roi_crops_evaluated": 0,
            "recovered_micro_vehicles": 0,
        }

        if self._prev_frame is None or self._prev_frame.shape != frame_curr.shape:
            self._prev_frame = frame_curr.copy()
            return recovered_detections, telemetry

        h, w = frame_curr.shape[:2]
        roadway_mask = self._ensure_roadway_mask((h, w), roadway_polygon)

        # 1. Extract roadway motion clusters
        current_rois = extract_roadway_motion_clusters(
            frame_curr=frame_curr,
            frame_prev=self._prev_frame,
            roadway_mask=roadway_mask,
            min_cluster_area=self.min_cluster_area,
            max_cluster_area=self.max_cluster_area,
        )
        self._prev_frame = frame_curr.copy()
        telemetry["motion_clusters_extracted"] = len(current_rois)

        # 2. Match with tracked clusters over time
        matched_cluster_ids = set()
        for roi in current_rois:
            rbox = roi[:4]
            rarea = roi[4]
            rcx = (rbox[0] + rbox[2]) / 2.0
            rcy = (rbox[1] + rbox[3]) / 2.0

            best_tc = None
            best_d = self.centroid_match_dist
            for tc in self._tracked_clusters:
                if tc["id"] in matched_cluster_ids:
                    continue
                tcx = (tc["box"][0] + tc["box"][2]) / 2.0
                tcy = (tc["box"][1] + tc["box"][3]) / 2.0
                d = float(np.hypot(rcx - tcx, rcy - tcy))
                if d < best_d:
                    best_d = d
                    best_tc = tc

            if best_tc is not None:
                best_tc["box"] = rbox
                best_tc["area"] = rarea
                best_tc["age"] += 1
                best_tc["last_frame"] = frame_idx
                matched_cluster_ids.add(best_tc["id"])
            else:
                self._tracked_clusters.append({
                    "id": self._next_cluster_id,
                    "box": rbox,
                    "area": rarea,
                    "age": 1,
                    "last_frame": frame_idx,
                    "active_hits": 0,
                    "last_eval_frame": -999,
                })
                self._next_cluster_id += 1

        # Prune stale clusters (missed for > 2 frames)
        self._tracked_clusters = [
            tc for tc in self._tracked_clusters
            if (frame_idx - tc["last_frame"]) <= 2
        ]
        telemetry["persistent_clusters_tracked"] = len([
            tc for tc in self._tracked_clusters if tc["age"] >= self.persistence_threshold
        ])

        # 3. Invocation condition:
        # - Primary YOLO pass at 640p returns 0 vehicle detections overlapping the motion ROI
        # - Motion cluster persists across >= 4 consecutive frames
        primary_veh_boxes = [
            tuple(d["bbox"]) for d in primary_raw_detections
            if d.get("class_name", "").lower() in self.TARGET_VEHICLE_CLASSES
        ]

        candidate_clusters = []
        for tc in self._tracked_clusters:
            if tc["age"] < self.persistence_threshold:
                continue
            # Adaptive throttle: clusters with confirmed vehicle hits refresh every 2 frames;
            # clusters with 0 hits (potential shadow/glare) cool down for 10 frames.
            cooldown = 2 if tc.get("active_hits", 0) > 0 else 10
            if (frame_idx - tc.get("last_eval_frame", -999)) < cooldown:
                continue

            t_box = tc["box"]
            has_primary_detection = any(is_box_overlapping_cluster(t_box, pb) for pb in primary_veh_boxes)
            if not has_primary_detection:
                candidate_clusters.append(tc)

        if not candidate_clusters:
            return recovered_detections, telemetry

        # 4. Enforce compute budget: global crop interval and at most max_crops_per_frame (default 1)
        if (frame_idx - self._last_crop_frame) < self.min_crop_interval:
            return recovered_detections, telemetry

        # Prioritize clusters that have established active vehicle hits, then by largest motion area
        candidate_clusters.sort(
            key=lambda c: (c.get("active_hits", 0) > 0, c["area"]),
            reverse=True,
        )
        selected_clusters = candidate_clusters[: self.max_crops_per_frame]

        for cluster in selected_clusters:
            bx1, by1, bx2, by2 = cluster["box"]
            cw = bx2 - bx1
            ch = by2 - by1
            if cw < 15 or ch < 15:
                continue

            crop = frame_curr[by1:by2, bx1:bx2]
            crop_scaled = cv2.resize(crop, (self.crop_resolution, self.crop_resolution))

            cluster["last_eval_frame"] = frame_idx
            self._last_crop_frame = frame_idx
            telemetry["roi_crops_evaluated"] += 1
            zres = yolo_model.predict(
                source=crop_scaled,
                imgsz=self.crop_resolution,
                conf=self.crop_conf_threshold,
                device="cpu",
                verbose=False,
            )[0]

            has_crop_hit = False
            if zres.boxes is not None and len(zres.boxes) > 0:
                b_xyxy = zres.boxes.xyxy.cpu().numpy()
                b_conf = zres.boxes.conf.cpu().numpy()
                b_cls = zres.boxes.cls.cpu().numpy()

                sx = cw / float(self.crop_resolution)
                sy = ch / float(self.crop_resolution)

                for zb, zc, zcl in zip(b_xyxy, b_conf, b_cls):
                    cname = yolo_model.names[int(zcl)].lower()
                    if cname == "train":
                        cname = "truck"
                    if cname in self.TARGET_VEHICLE_CLASSES:
                        has_crop_hit = True
                        gx1 = float(bx1 + zb[0] * sx)
                        gy1 = float(by1 + zb[1] * sy)
                        gx2 = float(bx1 + zb[2] * sx)
                        gy2 = float(by1 + zb[3] * sy)

                        rec_det = {
                            "bbox": [gx1, gy1, gx2, gy2],
                            "conf": float(zc),
                            "class_id": int(zcl),
                            "class_name": "candidate_vehicle_recovered",
                            "original_class_name": cname,
                            "recovery_reason": "motion_guided_roi_zoom",
                            "audit_reclassified": True,
                        }
                        recovered_detections.append(rec_det)
                        telemetry["recovered_micro_vehicles"] += 1

            # Update cluster hit history
            if has_crop_hit:
                cluster["active_hits"] = cluster.get("active_hits", 0) + 1
            else:
                cluster["active_hits"] = max(0, cluster.get("active_hits", 0) - 1)

        return recovered_detections, telemetry

    def reset(self) -> None:
        """Reset internal frame cache and tracked clusters between videos."""
        self._prev_frame = None
        self._tracked_clusters.clear()
        self._last_crop_frame = -999
        self._next_cluster_id = 1
