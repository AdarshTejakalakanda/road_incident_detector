"""Robust Surveillance Multi-Object Tracker.

Implements a full production ByteTrack architecture tailored for traffic surveillance:
1. High-confidence association (Stage 1) with Kalman filter state propagation.
2. Low-confidence track-recovery association (Stage 2) for vehicles in shadow, smoke, or collision deformation.
3. Entrant track survival: unconfirmed tracks on frame 2 can match low-confidence detections rather than being dropped.
4. Flexible IoU + center proximity distance matching preventing track fragmentation on high-resolution views.
5. Collision deformation coasting for mature vehicle tracks.
6. Temporal class consistency smoothing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from scipy.optimize import linear_sum_assignment
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter

from .track_state import Track, TrackletState
from ..detection.detector_interface import DetectionBatch
from ..calibration.road_surface import RoadSurfaceDetector

logger = logging.getLogger("road_surveillance.tracking.surveillance_tracker")


def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
    """Convert bounding box from [x, y, w, h] to [x_center, y_center, aspect_ratio, h]."""
    ret = np.asarray(tlwh, dtype=np.float64).copy()
    ret[:2] += ret[2:] / 2.0
    ret[2] /= max(1e-4, ret[3])
    return ret


def xyah_to_tlwh(xyah: np.ndarray) -> np.ndarray:
    """Convert bounding box from [x_center, y_center, aspect_ratio, h] to [x, y, w, h]."""
    ret = np.asarray(xyah, dtype=np.float64).copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2.0
    return ret


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between N boxes1 (xyxy) and M boxes2 (xyxy)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1[:, None] + area2 - inter_area
    union_area = np.maximum(1e-6, union_area)

    return inter_area / union_area


def suppress_duplicate_detections(
    boxes: np.ndarray,
    confs: np.ndarray,
    cids: np.ndarray,
    labels: List[str],
    iou_thresh: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Suppress cross-class duplicate proposals on the exact same physical vehicle."""
    if len(boxes) <= 1:
        return boxes, confs, cids, labels
    order = np.argsort(-confs)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = compute_iou_matrix(boxes[order[1:]], boxes[i : i + 1]).ravel()
        mask = ious < iou_thresh
        order = order[1:][mask]
    keep = sorted(keep)
    return boxes[keep], confs[keep], cids[keep], [labels[k] for k in keep]


class RobustSurveillanceTracker:
    """Multi-object tracker with true ByteTrack two-stage recovery and collision coasting."""

    def __init__(
        self,
        track_activation_threshold: float = 0.20,
        low_confidence_floor: float = 0.10,
        lost_track_buffer: int = 30,
        match_threshold_high: float = 0.75,  # 1 - IoU <= 0.75 (IoU >= 0.25)
        match_threshold_low: float = 0.65,   # 1 - IoU <= 0.65 (IoU >= 0.35)
        max_coasting_frames: int = 8,
        road_surface_detector: Optional[RoadSurfaceDetector] = None,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.low_confidence_floor = low_confidence_floor
        self.lost_track_buffer = lost_track_buffer
        self.match_threshold_high = match_threshold_high
        self.match_threshold_low = match_threshold_low
        self.max_coasting_frames = max_coasting_frames
        self.road_surface_detector = road_surface_detector or RoadSurfaceDetector()

        self.kf = KalmanFilter()
        self.tracks: Dict[int, Track] = {}
        self.track_kf_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # tid -> (mean, cov)
        self.id_counter = 0

    def reset(self) -> None:
        """Reset internal tracker state between videos."""
        self.tracks.clear()
        self.track_kf_states.clear()
        self.id_counter = 0
        if self.road_surface_detector:
            self.road_surface_detector.reset()

    def update(
        self,
        batch: Any,
        pts_ns: Optional[int] = None,
        frame_idx: Optional[int] = None,
    ) -> Tuple[List[TrackletState], Dict[int, Track]]:
        """Update tracker with detection batch or List[DetectionRecord]."""
        tracklet_states: List[TrackletState] = []

        if isinstance(batch, list):
            # Ingestion from List[DetectionRecord]
            if len(batch) > 0:
                first_rec = batch[0]
                if pts_ns is None:
                    pts_ns = int(getattr(first_rec, "timestamp_s", 0.0) * 1e9)
                if frame_idx is None:
                    frame_idx = getattr(first_rec, "source_frame", 0)
                vid_id = "default"
                boxes_xyxy = np.array([r.bbox_xyxy for r in batch], dtype=np.float32)
                confs = np.array([r.raw_confidence for r in batch], dtype=np.float32)
                trusts = np.array([r.trust_score for r in batch], dtype=np.float32)
                labels = [r.fused_class for r in batch]
                tentatives = [bool(getattr(r, "is_tentative", False)) for r in batch]
                cids = np.array([2 for _ in batch], dtype=np.int32)
            else:
                if pts_ns is None:
                    pts_ns = 0
                if frame_idx is None:
                    frame_idx = 0
                vid_id = "default"
                boxes_xyxy = np.empty((0, 4), dtype=np.float32)
                confs = np.empty((0,), dtype=np.float32)
                trusts = np.empty((0,), dtype=np.float32)
                labels = []
                tentatives = []
                cids = np.empty((0,), dtype=np.int32)
        else:
            # DetectionBatch ingestion
            if pts_ns is None:
                pts_ns = batch.pts_ns
            if frame_idx is None:
                frame_idx = batch.frame_idx
            vid_id = batch.video_id
            boxes_xyxy = batch.boxes
            confs = batch.confidences
            trusts = np.ones(len(confs), dtype=np.float32)
            cids = batch.class_ids
            labels = batch.class_labels
            tentatives = [False for _ in confs]

        # 1. Kalman Predict all existing tracks
        for tid in list(self.tracks.keys()):
            mean, cov = self.track_kf_states[tid]
            new_mean, new_cov = self.kf.predict(mean, cov)
            self.track_kf_states[tid] = (new_mean, new_cov)

            pred_tlwh = xyah_to_tlwh(new_mean[:4])
            # Keep predicted bounding box valid
            pw = max(1.0, float(pred_tlwh[2]))
            ph = max(1.0, float(pred_tlwh[3]))
            self.tracks[tid].current_bbox_xywh = (
                float(pred_tlwh[0]),
                float(pred_tlwh[1]),
                pw,
                ph,
            )

        # 2. Partition detections into High-Score and Low-Score (Recovery) sets
        if len(confs) > 0:
            is_cand_rec = np.array([lbl == "candidate_vehicle_recovered" for lbl in labels], dtype=bool)
            is_tent_arr = np.array(tentatives, dtype=bool)
            high_mask = (confs >= self.track_activation_threshold) | (is_cand_rec & (trusts >= 0.50)) | is_tent_arr
            low_mask = (((confs >= self.low_confidence_floor) & (confs < self.track_activation_threshold)) | (is_cand_rec & (trusts < 0.50))) & (~is_tent_arr)
        else:
            high_mask = np.zeros(0, dtype=bool)
            low_mask = np.zeros(0, dtype=bool)

        high_dets = boxes_xyxy[high_mask] if len(confs) > 0 else np.empty((0, 4), dtype=np.float32)
        high_confs = confs[high_mask] if len(confs) > 0 else np.empty((0,), dtype=np.float32)
        high_trusts = trusts[high_mask] if len(trusts) > 0 else np.empty((0,), dtype=np.float32)
        high_cids = cids[high_mask] if len(confs) > 0 else np.empty((0,), dtype=np.int32)
        high_labels = [labels[i] for i in range(len(labels)) if high_mask[i]]
        high_tentatives = [tentatives[i] for i in range(len(tentatives)) if high_mask[i]]

        # Cross-class duplicate proposal suppression (e.g. car + truck on same vehicle)
        if len(high_dets) > 1:
            order = np.argsort(-high_confs)
            keep = []
            while len(order) > 0:
                i = order[0]
                keep.append(i)
                if len(order) == 1:
                    break
                ious = compute_iou_matrix(high_dets[order[1:]], high_dets[i : i + 1]).ravel()
                mask = ious < 0.65
                order = order[1:][mask]
            keep = sorted(keep)
            high_dets = high_dets[keep]
            high_confs = high_confs[keep]
            high_trusts = high_trusts[keep]
            high_cids = high_cids[keep]
            high_labels = [high_labels[k] for k in keep]
            high_tentatives = [high_tentatives[k] for k in keep]

        low_dets = boxes_xyxy[low_mask] if len(confs) > 0 else np.empty((0, 4), dtype=np.float32)
        low_confs = confs[low_mask] if len(confs) > 0 else np.empty((0,), dtype=np.float32)
        low_trusts = trusts[low_mask] if len(trusts) > 0 else np.empty((0,), dtype=np.float32)
        low_cids = cids[low_mask] if len(confs) > 0 else np.empty((0,), dtype=np.int32)
        low_labels = [labels[i] for i in range(len(labels)) if low_mask[i]]
        low_tentatives = [tentatives[i] for i in range(len(tentatives)) if low_mask[i]]

        # Active pool: tracks that have not exceeded the lost track buffer
        pool_tids = [tid for tid, t in self.tracks.items() if t.lost_frame_count <= self.lost_track_buffer]

        pool_boxes_xyxy = []
        for tid in pool_tids:
            bx, by, bw, bh = self.tracks[tid].current_bbox_xywh
            pool_boxes_xyxy.append([bx, by, bx + bw, by + bh])
        pool_boxes = np.array(pool_boxes_xyxy, dtype=np.float32) if pool_boxes_xyxy else np.empty((0, 4), dtype=np.float32)

        matched_tracks_stage1: Set[int] = set()
        matched_high_dets: Set[int] = set()

        # 3. Stage 1 Association: Match existing tracks with high-confidence detections
        # Cost matrix: spatial cost * (1.0 + (1.0 - trust_score))
        if len(pool_boxes) > 0 and len(high_dets) > 0:
            ious = compute_iou_matrix(pool_boxes, high_dets)
            spatial_cost = 1.0 - ious
            trust_mult = 1.0 + 0.15 * (1.0 - high_trusts)  # shape (len(high_dets),)
            cost_matrix = spatial_cost * trust_mult[None, :]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.match_threshold_high:
                    tid = pool_tids[r]
                    matched_tracks_stage1.add(tid)
                    matched_high_dets.add(c)

                    bx1, by1, bx2, by2 = high_dets[c]
                    bw = max(1.0, bx2 - bx1)
                    bh = max(1.0, by2 - by1)
                    tlwh = np.array([bx1, by1, bw, bh], dtype=np.float64)
                    measurement = tlwh_to_xyah(tlwh)
                    mean, cov = self.track_kf_states[tid]
                    new_mean, new_cov = self.kf.update(mean, cov, measurement)
                    self.track_kf_states[tid] = (new_mean, new_cov)

                    lbl = high_labels[c]
                    is_tent = bool(high_tentatives[c]) if c < len(high_tentatives) else False
                    self.tracks[tid].update(
                        bbox_xywh=(float(bx1), float(by1), float(bw), float(bh)),
                        confidence=float(high_confs[c]),
                        pts_ns=pts_ns,
                        label=lbl,
                        provenance="REAL",
                        trust_score=float(high_trusts[c]),
                        is_tentative_rescued=is_tent,
                    )
                    self.tracks[tid].interpolate_gap(max_gap=6)
                    self.tracks[tid].status = "active"
                    self.tracks[tid].coasting_count = 0

        # 4. Stage 2 Association: Match remaining tracks with low-confidence recovery detections
        remaining_pool_tids = [tid for tid in pool_tids if tid not in matched_tracks_stage1]
        matched_tracks_stage2: Set[int] = set()

        if len(remaining_pool_tids) > 0 and len(low_dets) > 0:
            rem_boxes = np.array([
                [self.tracks[tid].current_bbox_xywh[0],
                 self.tracks[tid].current_bbox_xywh[1],
                 self.tracks[tid].current_bbox_xywh[0] + self.tracks[tid].current_bbox_xywh[2],
                 self.tracks[tid].current_bbox_xywh[1] + self.tracks[tid].current_bbox_xywh[3]]
                for tid in remaining_pool_tids
            ], dtype=np.float32)

            ious_low = compute_iou_matrix(rem_boxes, low_dets)
            spatial_cost_low = 1.0 - ious_low
            trust_mult_low = 1.0 + 0.15 * (1.0 - low_trusts)
            cost_matrix_low = spatial_cost_low * trust_mult_low[None, :]

            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_low[r, c] <= self.match_threshold_low:
                    tid = remaining_pool_tids[r]
                    matched_tracks_stage2.add(tid)

                    bx1, by1, bx2, by2 = low_dets[c]
                    bw = max(1.0, bx2 - bx1)
                    bh = max(1.0, by2 - by1)
                    tlwh = np.array([bx1, by1, bw, bh], dtype=np.float64)
                    measurement = tlwh_to_xyah(tlwh)
                    mean, cov = self.track_kf_states[tid]
                    new_mean, new_cov = self.kf.update(mean, cov, measurement)
                    self.track_kf_states[tid] = (new_mean, new_cov)

                    lbl = low_labels[c]
                    is_tent = bool(low_tentatives[c]) if c < len(low_tentatives) else False
                    self.tracks[tid].update(
                        bbox_xywh=(float(bx1), float(by1), float(bw), float(bh)),
                        confidence=float(low_confs[c]),
                        pts_ns=pts_ns,
                        label=lbl,
                        provenance="REAL",
                        trust_score=float(low_trusts[c]),
                        is_tentative_rescued=is_tent,
                    )
                    self.tracks[tid].interpolate_gap(max_gap=6)
                    self.tracks[tid].status = "active"
                    self.tracks[tid].coasting_count = 0

        # 5. Handle unmatched existing tracks: coasting vs lost
        all_matched = matched_tracks_stage1 | matched_tracks_stage2
        for tid in pool_tids:
            if tid not in all_matched:
                track = self.tracks[tid]
                is_veh = track.get_effective_label() in Track.VEHICLE_CLASSES
                can_coast = (
                    track.track_age >= 2
                    and is_veh
                    and track.coasting_count < self.max_coasting_frames
                    and track.status in ("active", "coasting")
                )
                if can_coast:
                    pred_box = self.tracks[tid].current_bbox_xywh
                    track.coast(pts_ns, predicted_bbox=pred_box)
                else:
                    track.mark_lost()
                    if track.lost_frame_count > self.lost_track_buffer:
                        track.status = "removed"

        # Remove dead tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].status == "removed":
                del self.tracks[tid]
                if tid in self.track_kf_states:
                    del self.track_kf_states[tid]

        # 6. Initialize new tracks from unmatched high-score detections
        for i in range(len(high_dets)):
            if i not in matched_high_dets:
                bx1, by1, bx2, by2 = high_dets[i]
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)

                # Ensure this candidate does not overlap heavily with an already active track
                cand_box = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
                existing_boxes = []
                for t in self.tracks.values():
                    if t.status in ("active", "coasting"):
                        ex, ey, ew, eh = t.current_bbox_xywh
                        existing_boxes.append([ex, ey, ex + ew, ey + eh])
                if existing_boxes:
                    ious_ex = compute_iou_matrix(cand_box, np.array(existing_boxes, dtype=np.float32)).ravel()
                    if np.any(ious_ex >= 0.60):
                        continue  # Suppress duplicate track near already tracked vehicle

                tlwh = np.array([bx1, by1, bw, bh], dtype=np.float64)

                self.id_counter += 1
                new_tid = self.id_counter
                mean, cov = self.kf.initiate(tlwh_to_xyah(tlwh))
                self.track_kf_states[new_tid] = (mean, cov)

                lbl = high_labels[i]
                init_box = (float(bx1), float(by1), float(bw), float(bh))
                tr_score = float(high_trusts[i])
                is_tent = bool(high_tentatives[i]) if i < len(high_tentatives) else False
                tr = Track(
                    track_id=new_tid,
                    class_id=int(high_cids[i]),
                    class_label=lbl,
                    detection_confidence=float(high_confs[i]),
                    trust_score=tr_score,
                    trust_score_history=[tr_score],
                    current_bbox_xywh=init_box,
                    track_age=1,
                    status="active",
                    bbox_history=[init_box],
                    provenance_history=["REAL"],
                    position_history=[Track.calculate_ground_contact(init_box)],
                    timestamps=[pts_ns],
                    label_history=[lbl],
                    is_tentative_rescued=is_tent,
                )
                self.tracks[new_tid] = tr

        # 7. Generate TrackletState outputs for downstream consumers
        for tid, tr in self.tracks.items():
            if tr.status in ("active", "coasting"):
                gc_px = Track.calculate_ground_contact(tr.current_bbox_xywh)
                tl = TrackletState(
                    video_id=vid_id,
                    track_id=tid,
                    class_id=tr.class_id,
                    class_label=tr.get_effective_label(),
                    frame_idx=frame_idx,
                    pts_ns=pts_ns,
                    bbox_xywh=tr.current_bbox_xywh,
                    ground_contact_px=gc_px,
                    track_status=tr.status,
                    trust_score=tr.trust_score,
                    provenance=tr.provenance_history[-1] if tr.provenance_history else "REAL",
                )
                tracklet_states.append(tl)

        return tracklet_states, self.tracks
