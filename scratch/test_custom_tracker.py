"""Test custom robust ByteTrack implementation on UarP8qU1S-c_00."""

import os
import sys
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))
from ultralytics import YOLO
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter
from road_surveillance.tracking.track_state import Track, TrackletState
from road_surveillance.calibration.road_surface import RoadSurfaceDetector

def tlwh_to_xyah(tlwh):
    ret = np.asarray(tlwh, dtype=np.float64).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

def xyah_to_tlwh(xyah):
    ret = np.asarray(xyah, dtype=np.float64).copy()
    ret[2] *= ret[3]
    ret[:2] -= ret[2:] / 2
    return ret

def compute_ious(boxes1, boxes2):
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


class RobustSurveillanceTracker:
    """Production-grade multi-object surveillance tracker with 2-stage low-confidence recovery."""

    def __init__(
        self,
        high_threshold: float = 0.20,
        low_threshold: float = 0.10,
        max_lost_frames: int = 30,
        match_thresh_high: float = 0.75, # 1 - IoU <= 0.75 (IoU >= 0.25)
        match_thresh_low: float = 0.65,  # 1 - IoU <= 0.65 (IoU >= 0.35)
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_lost_frames = max_lost_frames
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low

        self.kf = KalmanFilter()
        self.tracks: Dict[int, Track] = {}
        self.track_kf_states: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # tid -> (mean, cov)
        self.id_counter = 0

    def reset(self):
        self.tracks.clear()
        self.track_kf_states.clear()
        self.id_counter = 0

    def update(
        self,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        class_labels: List[str],
        pts_ns: int,
        frame_shape: Tuple[int, int],
    ) -> Dict[int, Track]:
        # 1. Predict all existing tracks
        for tid in list(self.tracks.keys()):
            mean, cov = self.track_kf_states[tid]
            new_mean, new_cov = self.kf.predict(mean, cov)
            self.track_kf_states[tid] = (new_mean, new_cov)

            # Update predicted bounding box
            pred_tlwh = xyah_to_tlwh(new_mean[:4])
            self.tracks[tid].current_bbox_xywh = (
                float(pred_tlwh[0]),
                float(pred_tlwh[1]),
                float(pred_tlwh[2]),
                float(pred_tlwh[3]),
            )

        # 2. Partition detections into High and Low score sets
        if len(confidences) > 0:
            high_mask = confidences >= self.high_threshold
            low_mask = (confidences >= self.low_threshold) & (confidences < self.high_threshold)
        else:
            high_mask = np.zeros(0, dtype=bool)
            low_mask = np.zeros(0, dtype=bool)

        high_dets = boxes_xyxy[high_mask] if len(confidences) > 0 else np.empty((0, 4))
        high_confs = confidences[high_mask] if len(confidences) > 0 else np.empty((0,))
        high_cids = class_ids[high_mask] if len(confidences) > 0 else np.empty((0,), dtype=int)
        high_labels = [class_labels[i] for i in range(len(class_labels)) if high_mask[i]]

        low_dets = boxes_xyxy[low_mask] if len(confidences) > 0 else np.empty((0, 4))
        low_confs = confidences[low_mask] if len(confidences) > 0 else np.empty((0,))
        low_cids = class_ids[low_mask] if len(confidences) > 0 else np.empty((0,), dtype=int)
        low_labels = [class_labels[i] for i in range(len(class_labels)) if low_mask[i]]

        # Active pool: all active or lost tracks (within max_lost_frames)
        pool_tids = [tid for tid, t in self.tracks.items() if t.lost_frame_count <= self.max_lost_frames]

        # Convert track predicted boxes to xyxy
        pool_boxes_xyxy = []
        for tid in pool_tids:
            bx, by, bw, bh = self.tracks[tid].current_bbox_xywh
            pool_boxes_xyxy.append([bx, by, bx + bw, by + bh])
        pool_boxes = np.array(pool_boxes_xyxy, dtype=np.float32) if pool_boxes_xyxy else np.empty((0, 4))

        matched_tracks_stage1 = set()
        matched_high_dets = set()

        # 3. Stage 1: Match pool with high score detections
        if len(pool_boxes) > 0 and len(high_dets) > 0:
            ious = compute_ious(pool_boxes, high_dets)
            cost_matrix = 1.0 - ious

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.match_thresh_high:
                    tid = pool_tids[r]
                    matched_tracks_stage1.add(tid)
                    matched_high_dets.add(c)

                    # Update Kalman filter
                    bx1, by1, bx2, by2 = high_dets[c]
                    tlwh = np.array([bx1, by1, bx2 - bx1, by2 - by1], dtype=np.float64)
                    measurement = tlwh_to_xyah(tlwh)
                    mean, cov = self.track_kf_states[tid]
                    new_mean, new_cov = self.kf.update(mean, cov, measurement)
                    self.track_kf_states[tid] = (new_mean, new_cov)

                    # Update Track
                    self.tracks[tid].update(
                        bbox_xywh=(float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])),
                        confidence=float(high_confs[c]),
                        pts_ns=pts_ns,
                        label=high_labels[c],
                    )

        # 4. Stage 2: Match remaining pool tracks with LOW score detections
        remaining_pool_tids = [tid for tid in pool_tids if tid not in matched_tracks_stage1]
        matched_tracks_stage2 = set()

        if len(remaining_pool_tids) > 0 and len(low_dets) > 0:
            rem_boxes = np.array([
                [self.tracks[tid].current_bbox_xywh[0],
                 self.tracks[tid].current_bbox_xywh[1],
                 self.tracks[tid].current_bbox_xywh[0] + self.tracks[tid].current_bbox_xywh[2],
                 self.tracks[tid].current_bbox_xywh[1] + self.tracks[tid].current_bbox_xywh[3]]
                for tid in remaining_pool_tids
            ], dtype=np.float32)

            ious_low = compute_ious(rem_boxes, low_dets)
            cost_matrix_low = 1.0 - ious_low

            row_ind, col_ind = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix_low[r, c] <= self.match_thresh_low:
                    tid = remaining_pool_tids[r]
                    matched_tracks_stage2.add(tid)

                    # Update Kalman filter
                    bx1, by1, bx2, by2 = low_dets[c]
                    tlwh = np.array([bx1, by1, bx2 - bx1, by2 - by1], dtype=np.float64)
                    measurement = tlwh_to_xyah(tlwh)
                    mean, cov = self.track_kf_states[tid]
                    new_mean, new_cov = self.kf.update(mean, cov, measurement)
                    self.track_kf_states[tid] = (new_mean, new_cov)

                    # Update Track
                    self.tracks[tid].update(
                        bbox_xywh=(float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])),
                        confidence=float(low_confs[c]),
                        pts_ns=pts_ns,
                        label=low_labels[c],
                    )

        # 5. Handle unmatched existing tracks: coasting vs lost
        all_matched = matched_tracks_stage1 | matched_tracks_stage2
        for tid in pool_tids:
            if tid not in all_matched:
                track = self.tracks[tid]
                is_veh = track.get_effective_label() in Track.VEHICLE_CLASSES
                can_coast = (track.track_age >= 2 and is_veh and track.coasting_count < 8 and track.status in ("active", "coasting"))
                if can_coast:
                    track.coast(pts_ns)
                else:
                    track.mark_lost()
                    if track.lost_frame_count > self.max_lost_frames:
                        track.status = "removed"

        # Remove dead tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].status == "removed":
                del self.tracks[tid]
                if tid in self.track_kf_states:
                    del self.track_kf_states[tid]

        # 6. Initialize new tracks from unmatched high score detections
        for i in range(len(high_dets)):
            if i not in matched_high_dets:
                bx1, by1, bx2, by2 = high_dets[i]
                bw = bx2 - bx1
                bh = by2 - by1
                tlwh = np.array([bx1, by1, bw, bh], dtype=np.float64)

                self.id_counter += 1
                new_tid = self.id_counter
                mean, cov = self.kf.initiate(tlwh_to_xyah(tlwh))
                self.track_kf_states[new_tid] = (mean, cov)

                tr = Track(
                    track_id=new_tid,
                    class_id=int(high_cids[i]),
                    class_label=high_labels[i],
                    detection_confidence=float(high_confs[i]),
                    current_bbox_xywh=(float(bx1), float(by1), float(bw), float(bh)),
                    track_age=1,
                    status="active",
                    position_history=[Track.calculate_ground_contact((float(bx1), float(by1), float(bw), float(bh)))],
                    timestamps=[pts_ns],
                    label_history=[high_labels[i]],
                )
                self.tracks[new_tid] = tr

        return self.tracks

if __name__ == "__main__":
    model = YOLO("yolo26n.pt")
    cap = cv2.VideoCapture(r"C:\Users\KALAKA~1\Downloads\archive\real_videos\UarP8qU1S-c_00.mp4")
    tracker = RobustSurveillanceTracker(high_threshold=0.20, low_threshold=0.10)

    f = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        p = model.predict(frame, conf=0.10, verbose=False)[0]
        b = p.boxes.xyxy.cpu().numpy()
        c = p.boxes.conf.cpu().numpy()
        cl = p.boxes.cls.cpu().numpy().astype(int)
        names = [model.names[x] for x in cl]

        tracks = tracker.update(b, c, cl, names, f * 33333333, frame.shape)
        if 98 <= f <= 135:
            active_tids = [tid for tid, tr in tracks.items() if tr.status in ("active", "coasting")]
            print(f"f={f:3d} (rel {f-130:+2d}): raw_dets={len(b)} active_tracks={len(active_tids)} {active_tids}")
            for tid in active_tids:
                tr = tracks[tid]
                print(f"   Track {tid}: status={tr.status} age={tr.track_age} coast={tr.coasting_count} label={tr.get_effective_label()} box={[round(x, 1) for x in tr.current_bbox_xywh]}")

        f += 1
    cap.release()
