import sys
import os
sys.path.insert(0, os.path.abspath("."))
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set

from road_surveillance.config import CollisionConfig
from road_surveillance.tracking import Track
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.events.candidates.distant_conflict import DistantTrajectoryConflictDetector
from scripts.benchmark_iteration_b_detector_recovery import (
    compute_iou,
    cluster_episodes,
    evaluate_video_candidates_temporal,
    RestoredIncidentRegion,
)

# Helper to compute pixel kinematics from Track history
def get_pixel_kinematics(tr: Track, pts_ns: int) -> Tuple[float, float]:
    if pts_ns in tr.timestamps:
        idx = tr.timestamps.index(pts_ns)
        pos_hist = tr.position_history[:idx+1]
        time_hist = tr.timestamps[:idx+1]
    else:
        pos_hist = tr.position_history
        time_hist = tr.timestamps
    px_spd = 0.0
    px_acc = 0.0
    if len(pos_hist) >= 2 and len(time_hist) >= 2:
        dt = (time_hist[-1] - time_hist[-2]) * 1e-9
        if dt > 1e-5:
            dp = np.hypot(pos_hist[-1][0] - pos_hist[-2][0], pos_hist[-1][1] - pos_hist[-2][1])
            px_spd = dp / dt
            if len(pos_hist) >= 3 and len(time_hist) >= 3:
                dt_prev = (time_hist[-2] - time_hist[-3]) * 1e-9
                if dt_prev > 1e-5:
                    dp_prev = np.hypot(pos_hist[-2][0] - pos_hist[-3][0], pos_hist[-2][1] - pos_hist[-3][1])
                    px_spd_prev = dp_prev / dt_prev
                    px_acc = (px_spd - px_spd_prev) / dt
    return px_spd, px_acc


class RefinedIncidentLifecycleManager:
    """Refined Lifecycle Manager using image-space coordinates and pixel kinematics for uncalibrated video."""

    def __init__(self, is_camera_calibrated: bool = False):
        self.is_camera_calibrated = is_camera_calibrated
        self.incidents: List[RestoredIncidentRegion] = []
        self.next_id = 1

    def reset(self):
        self.incidents.clear()
        self.next_id = 1

    def evaluate_candidate_interaction(
        self,
        frame_idx: int,
        t1_id: int,
        t2_id: int,
        tr1: Track,
        tr2: Track,
        k1: TrackKinematics,
        k2: TrackKinematics,
        px_centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[str]]:
        for inc in self.incidents:
            if inc.state == "CLEARED":
                continue

            # Centroid distance in image pixels
            c_dist = np.hypot(px_centroid[0] - inc.centroid[0], px_centroid[1] - inc.centroid[1])
            box_iou = compute_iou(interaction_bbox, inc.bbox)

            spatial_radius = 8.0 if self.is_camera_calibrated else max(150.0, 2.0 * max(inc.bbox[2], inc.bbox[3]))
            center_inside = (inc.bbox[0] <= px_centroid[0] <= inc.bbox[0] + inc.bbox[2]) and (
                inc.bbox[1] <= px_centroid[1] <= inc.bbox[1] + inc.bbox[3]
            )
            is_in_zone = (c_dist <= spatial_radius) or (box_iou >= 0.05) or center_inside

            if is_in_zone:
                t1_in_inc = (t1_id in inc.involved_track_ids) or (t1_id in inc.resting_track_ids)
                t2_in_inc = (t2_id in inc.involved_track_ids) or (t2_id in inc.resting_track_ids)

                # Check speed using pixel kinematics or calibrated speed
                if self.is_camera_calibrated:
                    s1_low = k1.speed_kmh < 6.0
                    s2_low = k2.speed_kmh < 6.0
                else:
                    px_spd1, _ = get_pixel_kinematics(tr1, k1.pts_ns)
                    px_spd2, _ = get_pixel_kinematics(tr2, k2.pts_ns)
                    char1 = max(15.0, max(tr1.current_bbox_xywh[2], tr1.current_bbox_xywh[3]))
                    char2 = max(15.0, max(tr2.current_bbox_xywh[2], tr2.current_bbox_xywh[3]))
                    s1_low = (px_spd1 < 35.0) or ((px_spd1 / char1) < 0.4)
                    s2_low = (px_spd2 < 35.0) or ((px_spd2 / char2) < 0.4)

                # 1. Damaged / Resting Wreckage Jitter
                if (t1_in_inc and t2_in_inc) or (s1_low and s2_low):
                    inc.resting_track_ids.add(t1_id)
                    inc.resting_track_ids.add(t2_id)
                    inc.state = "POST_IMPACT_RESTING"
                    inc.last_active_frame = frame_idx
                    inc.update_bounds(interaction_bbox)
                    return True, "WRECKAGE_RESTING"

                # 2. Passing Vehicle Grazing Stationary Wreckage
                if (t1_in_inc and not t2_in_inc) or (t2_in_inc and not t1_in_inc):
                    moving_tr = tr2 if t1_in_inc else tr1
                    moving_k = k2 if t1_in_inc else k1
                    moving_in_inc = t2_in_inc if t1_in_inc else t1_in_inc
                    inc.last_active_frame = frame_idx
                    return True, "PASSING_VEHICLE_RESTING_INTERACTION"

                # 3. Same incident spatial zone within post-impact window
                if (frame_idx - inc.origin_frame) <= 800:
                    inc.last_active_frame = frame_idx
                    inc.update_bounds(interaction_bbox)
                    return True, "POST_IMPACT_SPATIAL_ZONE"

        return False, None

    def register_incident(
        self,
        frame_idx: int,
        t1_id: int,
        t2_id: int,
        px_centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ):
        bx, by, bw, bh = interaction_bbox
        mw = bw * 0.4
        mh = bh * 0.4
        expanded = (max(0.0, bx - mw), max(0.0, by - mh), bw + 2 * mw, bh + 2 * mh)

        inc = RestoredIncidentRegion(
            region_id=self.next_id,
            origin_frame=frame_idx,
            centroid=px_centroid,
            bbox=expanded,
            involved_track_ids={t1_id, t2_id},
        )
        self.incidents.append(inc)
        self.next_id += 1


class RefinedIterationBCollisionDetector:
    """Refined Detector B: Image-space pixel proximity and pixel kinematics gating."""

    def __init__(self, is_camera_calibrated: bool = False, config: Optional[CollisionConfig] = None):
        self.is_camera_calibrated = is_camera_calibrated
        self.cfg = config or CollisionConfig()
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_ious: Dict[Tuple[int, int], float] = {}
        self.spatial_clusters: List[Dict[str, Any]] = []
        self.lifecycle_mgr = RefinedIncidentLifecycleManager(is_camera_calibrated=is_camera_calibrated)

    def reset(self):
        self.persistence_counts.clear()
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.spatial_clusters.clear()
        self.lifecycle_mgr.reset()

    def evaluate_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
    ) -> List[Dict]:
        candidates = []
        current_pair_keys = set()
        VEHICLES = {"car", "truck", "bus", "van", "motorcycle"}

        for conv in convergences:
            t1_id, t2_id = conv.track_id_1, conv.track_id_2
            k1, k2 = kinematics.get(t1_id), kinematics.get(t2_id)
            tr1, tr2 = tracks.get(t1_id), tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            # Ground coordinates validity check
            gx1, gy1 = k1.ground_pos
            gx2, gy2 = k2.ground_pos
            g_valid1 = (abs(gx1) <= 150.0 and -50.0 <= gy1 <= 250.0)
            g_valid2 = (abs(gx2) <= 150.0 and -50.0 <= gy2 <= 250.0)

            # Bboxes and pixel distances
            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            iou = compute_iou(b1, b2)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            c1 = (b1[0] + b1[2]/2.0, b1[1] + b1[3]/2.0)
            c2 = (b2[0] + b2[2]/2.0, b2[1] + b2[3]/2.0)
            px_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_px_dist = px_dist / char_dim

            # Track maturity
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            # Pixel kinematics
            px_spd1, px_acc1 = get_pixel_kinematics(tr1, pts_ns)
            px_spd2, px_acc2 = get_pixel_kinematics(tr2, pts_ns)
            norm_px_acc1 = (px_acc1 / max(15.0, max(b1[2], b1[3]))) * 4.5
            norm_px_acc2 = (px_acc2 / max(15.0, max(b2[2], b2[3]))) * 4.5

            # 1. Proximity signal
            if self.is_camera_calibrated and g_valid1 and g_valid2:
                proximity_signal = (conv.distance_m <= self.cfg.max_ground_distance_m) or (
                    iou >= self.cfg.min_iou_overlap and conv.distance_m <= self.cfg.max_perspective_distance_m
                )
            else:
                # In uncalibrated mode, MUST satisfy image-space proximity
                # (either bounding box overlap or close center-to-center distance)
                proximity_signal = (iou >= self.cfg.min_iou_overlap) or (norm_px_dist <= 1.2 and px_dist <= 180.0)

            if not proximity_signal:
                continue

            # 2. Deceleration signal
            if self.is_camera_calibrated and g_valid1 and g_valid2:
                decel_signal = (k1.acceleration_mag <= self.cfg.deceleration_threshold_ms2) or (
                    k2.acceleration_mag <= self.cfg.deceleration_threshold_ms2
                )
            else:
                # In uncalibrated mode, use pixel-scaled acceleration OR metric acceleration if valid
                if g_valid1 and g_valid2:
                    raw_a1 = k1.acceleration_mag * (4.5 / max(15.0, max(b1[2], b1[3])))
                    raw_a2 = k2.acceleration_mag * (4.5 / max(15.0, max(b2[2], b2[3])))
                    decel_signal = (raw_a1 <= self.cfg.deceleration_threshold_ms2) or (
                        raw_a2 <= self.cfg.deceleration_threshold_ms2
                    ) or (norm_px_acc1 <= self.cfg.deceleration_threshold_ms2) or (
                        norm_px_acc2 <= self.cfg.deceleration_threshold_ms2
                    )
                else:
                    # Ground invalid: rely strictly on pixel acceleration
                    decel_signal = (norm_px_acc1 <= self.cfg.deceleration_threshold_ms2) or (
                        norm_px_acc2 <= self.cfg.deceleration_threshold_ms2
                    ) or (iou >= 0.20) # high visual overlap can trigger impact

            # 3. Convergence signal
            prev_d = self.prev_distances.get(pair_key, px_dist)
            is_closing = (px_dist - prev_d) < -1.0
            self.prev_distances[pair_key] = px_dist
            conv_signal = bool(is_closing or iou_increasing or (iou >= self.cfg.min_iou_overlap))

            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))
            px_centroid = (ux1 + (ux2 - ux1)/2.0, uy1 + (uy2 - uy1)/2.0)

            # Lifecycle gating
            is_redundant, _ = self.lifecycle_mgr.evaluate_candidate_interaction(
                frame_idx, t1_id, t2_id, tr1, tr2, k1, k2, px_centroid, interaction_box
            )
            if is_redundant:
                continue

            cond = proximity_signal and conv_signal and decel_signal
            if cond:
                self.lifecycle_mgr.register_incident(
                    frame_idx, t1_id, t2_id, px_centroid, interaction_box
                )
                candidates.append({
                    "frame_idx": frame_idx,
                    "pts_ns": pts_ns,
                    "track_ids": [t1_id, t2_id],
                    "dist": float(conv.distance_m),
                    "iou": float(iou),
                    "accel_1": float(k1.acceleration_mag),
                    "accel_2": float(k2.acceleration_mag),
                })

        for pk in list(self.prev_distances.keys()):
            if pk not in current_pair_keys:
                del self.prev_distances[pk]
                self.prev_ious.pop(pk, None)
                self.persistence_counts.pop(pk, None)

        return candidates


print("Classes defined.")
