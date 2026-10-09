"""Iteration B: Selective Detector Recovery & Restored Incident Lifecycle Benchmark.

Implements Iteration B:
- Normal detection: confidence >= 0.25
- Recovery mode: confidence in [0.15, 0.25)
    ONLY when:
        existing mature track exists (track_age >= 2)
        AND another vehicle is in visual conflict (IoU / normalized proximity / closing)
- Calibration-aware 2D collision geometry (from Iteration A)
- Restored calibration-aware Incident Lifecycle Manager (pixel-scaled spatial envelope)
- Preserves +/-35 frames tolerance, separately reporting GT vs Predicted vs Physical frames.

Evaluates on BOTH:
- split_in_distribution (15 test videos)
- split_geo_aware (15 test videos)
"""

import os
import sys
import json
import pickle
import argparse
import time
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from road_surveillance.config import load_config, CollisionConfig
from road_surveillance.video import DatasetVideoIngestor, IngestedFrame
from road_surveillance.detection import YOLO26Detector, DetectionBatch
from road_surveillance.tracking import ByteTrackTracker, Track
from road_surveillance.calibration import PlanarHomography, LaneManager
from road_surveillance.geometry import GeometryEngine
from road_surveillance.geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from road_surveillance.evaluation import GroundTruthLoader


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute 2D bounding box Intersection over Union."""
    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter
    return float(inter / union) if union > 1e-6 else 0.0


def xyxy_to_xywh(box: np.ndarray) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


class SelectiveRecoveryDetector:
    """Wrapper that filters YOLO detections to selectively recover low-confidence detections during visual conflict."""

    def __init__(self, yolo_detector: YOLO26Detector, normal_threshold: float = 0.25, recovery_floor: float = 0.15):
        self.detector = yolo_detector
        self.normal_threshold = normal_threshold
        self.recovery_floor = recovery_floor

    def detect_selective(
        self,
        frame: IngestedFrame,
        current_tracks: Dict[int, Track],
    ) -> DetectionBatch:
        # Run YOLO with the lower floor threshold
        prev_conf = self.detector.conf_threshold
        self.detector.conf_threshold = self.recovery_floor
        batch = self.detector.detect(frame)
        self.detector.conf_threshold = prev_conf

        if len(batch.boxes) == 0:
            return batch

        # Identify tracks currently in visual conflict
        active_mature_tracks = [t for t in current_tracks.values() if t.status == "active" and t.track_age >= 2]
        conflict_tracks: Set[int] = set()

        for i, t1 in enumerate(active_mature_tracks):
            b1 = t1.current_bbox_xywh
            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            L1 = max(15.0, max(b1[2], b1[3]))

            for t2 in active_mature_tracks[i + 1:]:
                b2 = t2.current_bbox_xywh
                c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
                L2 = max(15.0, max(b2[2], b2[3]))

                iou = compute_iou(b1, b2)
                d_px = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                norm_d = d_px / ((L1 + L2) / 2.0)

                # Conflict condition: high overlap or close proximity
                if iou >= 0.05 or norm_d <= 1.5:
                    conflict_tracks.add(t1.track_id)
                    conflict_tracks.add(t2.track_id)

        # Selectively filter detections
        keep_indices = []
        for idx in range(len(batch.boxes)):
            conf = float(batch.confidences[idx])
            box_xywh = xyxy_to_xywh(batch.boxes[idx])

            if conf >= self.normal_threshold:
                # Normal detection: always keep
                keep_indices.append(idx)
            else:
                # Recovery candidate (0.15 <= conf < 0.25):
                # Must match an existing mature track in conflict
                recovered = False
                for tid in conflict_tracks:
                    tr = current_tracks.get(tid)
                    if not tr:
                        continue
                    iou = compute_iou(box_xywh, tr.current_bbox_xywh)
                    # Check center distance
                    c_det = (box_xywh[0] + box_xywh[2] / 2.0, box_xywh[1] + box_xywh[3] / 2.0)
                    c_tr = (tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2] / 2.0, tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3] / 2.0)
                    dist_px = np.hypot(c_det[0] - c_tr[0], c_det[1] - c_tr[1])
                    char_len = max(tr.current_bbox_xywh[2], tr.current_bbox_xywh[3])

                    if iou >= 0.20 or dist_px <= (0.8 * char_len):
                        recovered = True
                        break

                if recovered:
                    keep_indices.append(idx)

        if len(keep_indices) == len(batch.boxes):
            return batch

        if not keep_indices:
            return DetectionBatch(
                video_id=batch.video_id,
                frame_idx=batch.frame_idx,
                pts_ns=batch.pts_ns,
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        indices = np.array(keep_indices, dtype=np.int32)
        return DetectionBatch(
            video_id=batch.video_id,
            frame_idx=batch.frame_idx,
            pts_ns=batch.pts_ns,
            boxes=batch.boxes[indices],
            confidences=batch.confidences[indices],
            class_ids=batch.class_ids[indices],
            class_labels=[batch.class_labels[i] for i in keep_indices],
        )


class RestoredIncidentRegion:
    def __init__(self, region_id: int, origin_frame: int, centroid: Tuple[float, float], bbox: Tuple[float, float, float, float], involved_track_ids: Set[int]):
        self.region_id = region_id
        self.origin_frame = origin_frame
        self.centroid = centroid
        self.bbox = bbox
        self.involved_track_ids = set(involved_track_ids)
        self.resting_track_ids: Set[int] = set()
        self.state = "ACTIVE_COLLISION"
        self.last_active_frame = origin_frame

    def update_bounds(self, new_bbox: Tuple[float, float, float, float]):
        bx, by, bw, bh = self.bbox
        nx, ny, nw, nh = new_bbox
        ux1 = min(bx, nx)
        uy1 = min(by, ny)
        ux2 = max(bx + bw, nx + nw)
        uy2 = max(by + bh, ny + nh)
        self.bbox = (ux1, uy1, ux2 - ux1, uy2 - uy1)


class RestoredIncidentLifecycleManager:
    """Calibration-aware Incident Lifecycle Manager with pixel-scaled spatial matching."""

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
        centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[str]]:
        for inc in self.incidents:
            if inc.state == "CLEARED":
                continue

            c_dist = np.hypot(centroid[0] - inc.centroid[0], centroid[1] - inc.centroid[1])
            box_iou = compute_iou(interaction_bbox, inc.bbox)

            # Spatial threshold: 8.0 meters if calibrated, or proportional pixel radius (>= 150 px) if uncalibrated
            spatial_radius = 8.0 if self.is_camera_calibrated else max(150.0, 2.0 * max(inc.bbox[2], inc.bbox[3]))

            # Center inside bounding box or proximity / IoU
            center_inside = (inc.bbox[0] <= centroid[0] <= inc.bbox[0] + inc.bbox[2]) and (
                inc.bbox[1] <= centroid[1] <= inc.bbox[1] + inc.bbox[3]
            )
            is_in_zone = (c_dist <= spatial_radius) or (box_iou >= 0.05) or center_inside

            if is_in_zone:
                t1_in_inc = (t1_id in inc.involved_track_ids) or (t1_id in inc.resting_track_ids)
                t2_in_inc = (t2_id in inc.involved_track_ids) or (t2_id in inc.resting_track_ids)

                if self.is_camera_calibrated:
                    s1_low = k1.speed_kmh < 6.0
                    s2_low = k2.speed_kmh < 6.0
                else:
                    from road_surveillance.geometry.spatial_metrics import compute_pixel_kinematics
                    px_spd1, _ = compute_pixel_kinematics(tr1, k1.pts_ns)
                    px_spd2, _ = compute_pixel_kinematics(tr2, k2.pts_ns)
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
                    inc.last_active_frame = frame_idx
                    return True, "PASSING_VEHICLE_RESTING_INTERACTION"

                # 3. Post-impact spatial zone absorption within 800 frames
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
        centroid: Tuple[float, float],
        interaction_bbox: Tuple[float, float, float, float],
    ):
        bx, by, bw, bh = interaction_bbox
        mw = bw * 0.4
        mh = bh * 0.4
        expanded = (max(0.0, bx - mw), max(0.0, by - mh), bw + 2 * mw, bh + 2 * mh)

        inc = RestoredIncidentRegion(
            region_id=self.next_id,
            origin_frame=frame_idx,
            centroid=centroid,
            bbox=expanded,
            involved_track_ids={t1_id, t2_id},
        )
        self.incidents.append(inc)
        self.next_id += 1


class IterationBCollisionDetector:
    """Evaluates collision candidates with Calibration-Aware Geometry & Restored Lifecycle."""

    def __init__(
        self,
        is_camera_calibrated: bool = False,
        config: Optional[CollisionConfig] = None,
        max_physical_decel_ms2: float = 60.0,
    ):
        self.is_camera_calibrated = is_camera_calibrated
        self.cfg = config or CollisionConfig()
        self.max_physical_decel = max_physical_decel_ms2
        self.persistence_counts: Dict[Tuple[int, int], int] = {}
        self.prev_distances: Dict[Tuple[int, int], float] = {}
        self.prev_ious: Dict[Tuple[int, int], float] = {}
        self.spatial_clusters: List[Dict[str, Any]] = []
        self.lifecycle_mgr = RestoredIncidentLifecycleManager(is_camera_calibrated=is_camera_calibrated)

    def reset(self):
        self.persistence_counts.clear()
        self.prev_distances.clear()
        self.prev_ious.clear()
        self.spatial_clusters.clear()
        self.lifecycle_mgr.reset()

    def _estimate_metric_acceleration(self, track: Track, raw_accel: float) -> float:
        if self.is_camera_calibrated:
            return raw_accel
        w, h = track.current_bbox_xywh[2], track.current_bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

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
        VEHICLES = {"car", "truck", "bus", "van", "motorcycle", "train"}

        for conv in convergences:
            t1_id = conv.track_id_1
            t2_id = conv.track_id_2
            k1 = kinematics.get(t1_id)
            k2 = kinematics.get(t2_id)
            tr1 = tracks.get(t1_id)
            tr2 = tracks.get(t2_id)

            if not k1 or not k2 or not tr1 or not tr2:
                continue

            if tr1.class_label.lower() not in VEHICLES or tr2.class_label.lower() not in VEHICLES:
                continue

            pair_key = (min(t1_id, t2_id), max(t1_id, t2_id))
            current_pair_keys.add(pair_key)

            iou = compute_iou(tr1.current_bbox_xywh, tr2.current_bbox_xywh)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            # 1. Track Maturity Gate
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            b1 = tr1.current_bbox_xywh
            b2 = tr2.current_bbox_xywh
            iou = compute_iou(b1, b2)
            prev_iou = self.prev_ious.get(pair_key, iou)
            iou_increasing = (iou - prev_iou) > 0.02
            self.prev_ious[pair_key] = iou

            c1 = (b1[0] + b1[2] / 2.0, b1[1] + b1[3] / 2.0)
            c2 = (b2[0] + b2[2] / 2.0, b2[1] + b2[3] / 2.0)
            px_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
            char_dim = max(15.0, (max(b1[2], b1[3]) + max(b2[2], b2[3])) / 2.0)
            norm_px_dist = px_dist / char_dim

            # 1. Track Maturity Gate
            if tr1.track_age < 3 or tr2.track_age < 3:
                continue

            # Ground coordinates validity check
            gx1, gy1 = k1.ground_pos
            gx2, gy2 = k2.ground_pos
            g_valid1 = (abs(gx1) <= 150.0 and -50.0 <= gy1 <= 250.0)
            g_valid2 = (abs(gx2) <= 150.0 and -50.0 <= gy2 <= 250.0)

            # Pixel kinematics
            from road_surveillance.geometry.spatial_metrics import compute_pixel_kinematics
            px_spd1, px_acc1 = getattr(k1, "pixel_speed", 0.0), getattr(k1, "pixel_acceleration", 0.0)
            if px_spd1 == 0.0:
                px_spd1, px_acc1 = compute_pixel_kinematics(tr1, pts_ns)
            px_spd2, px_acc2 = getattr(k2, "pixel_speed", 0.0), getattr(k2, "pixel_acceleration", 0.0)
            if px_spd2 == 0.0:
                px_spd2, px_acc2 = compute_pixel_kinematics(tr2, pts_ns)

            norm_px_acc1 = (px_acc1 / max(15.0, max(b1[2], b1[3]))) * 4.5
            norm_px_acc2 = (px_acc2 / max(15.0, max(b2[2], b2[3]))) * 4.5

            # Spatial proximity check
            if self.is_camera_calibrated and g_valid1 and g_valid2:
                proximity_signal = (dist <= self.cfg.max_ground_distance_m) or (
                    iou >= self.cfg.min_iou_overlap and dist <= self.cfg.max_perspective_distance_m
                )
            else:
                # In uncalibrated mode, MUST satisfy image-space proximity
                # (rejects distant cross-highway pairs > 500 px / > 5.0 body-lengths apart)
                proximity_signal = (iou >= self.cfg.min_iou_overlap) or (norm_px_dist <= 2.5 and px_dist <= 450.0)

            if not proximity_signal:
                continue

            # Kinematic Deceleration signal
            raw_a1 = k1.acceleration_mag * (4.5 / max(15.0, max(b1[2], b1[3])))
            raw_a2 = k2.acceleration_mag * (4.5 / max(15.0, max(b2[2], b2[3])))
            decel_signal = (
                (-60.0 <= raw_a1 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= raw_a2 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= norm_px_acc1 <= self.cfg.deceleration_threshold_ms2) or
                (-60.0 <= norm_px_acc2 <= self.cfg.deceleration_threshold_ms2)
            )

            prev_d = self.prev_distances.get(pair_key, px_dist)
            is_closing = (px_dist - prev_d) < -1.0
            self.prev_distances[pair_key] = px_dist
            conv_signal = bool(is_closing or iou_increasing)

            ux1 = min(b1[0], b2[0])
            uy1 = min(b1[1], b2[1])
            ux2 = max(b1[0] + b1[2], b2[0] + b2[2])
            uy2 = max(b1[1] + b1[3], b2[1] + b2[3])
            interaction_box = (ux1, uy1, max(1.0, ux2 - ux1), max(1.0, uy2 - uy1))
            px_centroid = (ux1 + (ux2 - ux1) / 2.0, uy1 + (uy2 - uy1) / 2.0)

            # 5. Restored Incident Lifecycle Gating
            is_redundant, reason = self.lifecycle_mgr.evaluate_candidate_interaction(
                frame_idx, t1_id, t2_id, tr1, tr2, k1, k2, px_centroid, interaction_box
            )
            if is_redundant:
                continue

            cond = proximity_signal and conv_signal and decel_signal
            triggered = False

            if cond:
                matched = False
                for cluster in self.spatial_clusters:
                    if cluster["last_frame"] < frame_idx and (frame_idx - cluster["last_frame"]) <= 3:
                        box_iou = compute_iou(interaction_box, cluster["box"])
                        c_dist = np.hypot(px_centroid[0] - cluster["centroid"][0], px_centroid[1] - cluster["centroid"][1])
                        if box_iou >= 0.10 or c_dist <= 25.0:
                            cluster["persistence"] += 1
                            cluster["last_frame"] = frame_idx
                            cluster["centroid"] = px_centroid
                            cluster["box"] = interaction_box
                            matched = True
                            if cluster["persistence"] >= self.cfg.min_persistence_frames:
                                triggered = True
                            break
                if not matched:
                    self.spatial_clusters.append({
                        "centroid": px_centroid,
                        "box": interaction_box,
                        "last_frame": frame_idx,
                        "persistence": 1,
                    })
                    if self.cfg.min_persistence_frames <= 1:
                        triggered = True

            if triggered:
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

        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates


def cluster_episodes(raw_candidates: List[Dict], max_gap_frames: int = 25) -> List[Dict]:
    if not raw_candidates:
        return []

    sorted_cands = sorted(raw_candidates, key=lambda c: c["frame_idx"])
    episodes = []
    current_ep = [sorted_cands[0]]

    for c in sorted_cands[1:]:
        if (c["frame_idx"] - current_ep[-1]["frame_idx"]) <= max_gap_frames:
            current_ep.append(c)
        else:
            rep = current_ep[0]
            episodes.append({
                "start_frame": current_ep[0]["frame_idx"],
                "end_frame": current_ep[-1]["frame_idx"],
                "trigger_frame": current_ep[0]["frame_idx"],
                "raw_count": len(current_ep),
                "representative": rep,
            })
            current_ep = [c]

    if current_ep:
        rep = current_ep[0]
        episodes.append({
            "start_frame": current_ep[0]["frame_idx"],
            "end_frame": current_ep[-1]["frame_idx"],
            "trigger_frame": current_ep[0]["frame_idx"],
            "raw_count": len(current_ep),
            "representative": rep,
        })

    return episodes


def evaluate_video_candidates_temporal(
    raw_candidates: List[Dict],
    gt_frame: int,
    fps: float,
    tolerance_frames: int = 35,
    max_gap_frames: int = 25,
) -> Dict[str, Any]:
    episodes = cluster_episodes(raw_candidates, max_gap_frames=max_gap_frames)

    matched_episode = None
    pre_impact_fps = []
    post_impact_fps = []
    repeated_incident_fps = []
    raw_fp_count = 0

    for ep in episodes:
        tf = ep["trigger_frame"]
        diff = tf - gt_frame

        if abs(diff) <= tolerance_frames and matched_episode is None:
            matched_episode = ep
        else:
            raw_fp_count += ep["raw_count"]
            if diff < -tolerance_frames:
                pre_impact_fps.append(ep)
            else:
                post_impact_fps.append(ep)
                if diff > tolerance_frames and abs(diff) <= 800:
                    repeated_incident_fps.append(ep)

    is_tp = matched_episode is not None
    lead_frames = (gt_frame - matched_episode["trigger_frame"]) if is_tp else None
    lead_sec = (lead_frames / fps) if (is_tp and fps > 0) else None

    return {
        "is_tp": is_tp,
        "matched_episode": matched_episode,
        "lead_frames": lead_frames,
        "lead_sec": lead_sec,
        "total_episodes": len(episodes),
        "fp_episodes": len(pre_impact_fps) + len(post_impact_fps),
        "pre_impact_fp_count": len(pre_impact_fps),
        "post_impact_fp_count": len(post_impact_fps),
        "repeated_incident_fp_count": len(repeated_incident_fps),
        "raw_fp_count": raw_fp_count,
        "raw_candidates_count": len(raw_candidates),
    }


def ingest_with_recovery_or_load_cache(split_type: str, max_videos: int, cfg) -> List[Dict]:
    """Ingest videos using Selective Low-Confidence Recovery or load pre-cached data."""
    cache_file = f"cache_videos_iter_b_{split_type}_{max_videos}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading {split_type} from Iteration B cache: {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Generating Iteration B cache for {split_type} ({max_videos} videos)...")
    gt_loader = GroundTruthLoader(cfg.evaluation.metadata_path)
    records = gt_loader.filter_split(split="test", split_type=split_type)

    base_yolo = YOLO26Detector(cfg.detection.model_path, conf_threshold=0.25, device=cfg.detection.device)
    selective_det = SelectiveRecoveryDetector(base_yolo, normal_threshold=0.25, recovery_floor=0.15)
    tracker = ByteTrackTracker()

    video_data = []
    t0 = time.time()
    for idx, gt_rec in enumerate(records[:max_videos], 1):
        v_name = os.path.basename(gt_rec.video_path)
        v_path = os.path.join(cfg.dataset.path, v_name)
        if not os.path.exists(v_path):
            continue

        print(f"  [{idx}/{max_videos}] Ingesting with selective recovery: {gt_rec.video_id} (GT={gt_rec.accident_frame})...")
        ingestor = DatasetVideoIngestor(v_path)
        h = PlanarHomography()
        geom = GeometryEngine(h)
        tracker.reset()

        frame_records = []
        fps = 30.0

        for frame in ingestor.stream_video(v_path):
            fps = frame.fps
            batch = selective_det.detect_selective(frame, tracker.tracks)
            tl, tracks = tracker.update(batch)
            kin, conv = geom.process_frame(tl, tracks)
            frame_records.append((frame.frame_idx, frame.pts_ns, dict(tracks), dict(kin), list(conv)))

        video_data.append({
            "video_id": gt_rec.video_id,
            "gt_frame": gt_rec.accident_frame,
            "fps": fps,
            "frames": frame_records,
        })

    print(f"Completed in {time.time() - t0:.1f}s. Saving to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(video_data, f)

    return video_data


def run_benchmark():
    cfg = load_config("road_surveillance/config/default_config.yaml")

    exp_cfg = CollisionConfig(
        max_ground_distance_m=4.0,
        max_perspective_distance_m=12.0,
        min_iou_overlap=0.15,
        min_persistence_frames=2,
        temporal_cluster_gap_frames=25,
    )

    det_b = IterationBCollisionDetector(is_camera_calibrated=False, config=exp_cfg)

    all_split_results = {}
    detailed_timing_records = {}

    for split_type in ["split_in_distribution", "split_geo_aware"]:
        # 1. Load Iteration A baseline videos
        with open(f"cache_videos_{split_type}_15.pkl", "rb") as f:
            vids_a = pickle.load(f)

        # 2. Ingest or load Iteration B videos (with selective recovery)
        vids_b = ingest_with_recovery_or_load_cache(split_type, 15, cfg)

        print(f"\nEvaluating Iteration A vs Iteration B on '{split_type}'...")

        # We evaluate:
        # A: Iteration A (from previous benchmark)
        # B: Iteration B (Selective Recovery + Restored Lifecycle)
        results = {}
        timing_list = []

        for cid, clabel, vid_set in [
            ("iter_a", "Iteration A (Calibration-Aware)", vids_a),
            ("iter_b", "Iteration B (Selective Recovery + Restored Lifecycle)", vids_b),
        ]:
            tp_count = 0
            fn_count = 0
            fp_episodes_total = 0
            raw_fp_total = 0
            pre_fps_total = 0
            post_fps_total = 0
            rep_fps_total = 0
            lead_times_frames = []
            lead_times_sec = []
            abs_temp_errors = []

            for v in vid_set:
                gt_f = v["gt_frame"]
                fps = v["fps"]
                vid_id = v["video_id"]

                det_b.reset()
                cands = []
                for f_idx, pts_ns, tr, kin, conv in v["frames"]:
                    cands.extend(det_b.evaluate_frame(f_idx, pts_ns, tr, kin, conv))

                eval_res = evaluate_video_candidates_temporal(
                    cands, gt_f, fps, tolerance_frames=35, max_gap_frames=25
                )

                if eval_res["is_tp"]:
                    tp_count += 1
                    lead_times_frames.append(eval_res["lead_frames"])
                    lead_times_sec.append(eval_res["lead_sec"])
                    abs_temp_errors.append(abs(eval_res["lead_frames"]))
                    outcome = "TP"
                    pred_frame = eval_res["matched_episode"]["trigger_frame"]
                    err_f = eval_res["lead_frames"]
                    lead_s = eval_res["lead_sec"]
                else:
                    fn_count += 1
                    outcome = "FN"
                    pred_frame = None
                    err_f = None
                    lead_s = None

                fp_episodes_total += eval_res["fp_episodes"]
                raw_fp_total += eval_res["raw_fp_count"]
                pre_fps_total += eval_res["pre_impact_fp_count"]
                post_fps_total += eval_res["post_impact_fp_count"]
                rep_fps_total += eval_res["repeated_incident_fp_count"]

                if cid == "iter_b":
                    timing_list.append({
                        "video_id": vid_id,
                        "gt_frame": gt_f,
                        "pred_frame": pred_frame,
                        "temporal_error_frames": err_f,
                        "lead_time_sec": round(lead_s, 2) if lead_s is not None else None,
                        "outcome": outcome,
                    })

            n_vids = len(vid_set)
            prec = tp_count / (tp_count + fp_episodes_total) if (tp_count + fp_episodes_total) > 0 else 0.0
            rec = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            results[cid] = {
                "label": clabel,
                "tp": tp_count,
                "fp_episodes": fp_episodes_total,
                "raw_fp": raw_fp_total,
                "fn": fn_count,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fp_per_vid": fp_episodes_total / n_vids,
                "raw_fp_per_vid": raw_fp_total / n_vids,
                "pre_impact_fp": pre_fps_total,
                "post_impact_fp": post_fps_total,
                "repeated_fp": rep_fps_total,
                "med_lead_sec": float(np.median(lead_times_sec)) if lead_times_sec else 0.0,
                "med_lead_frames": float(np.median(lead_times_frames)) if lead_times_frames else 0.0,
                "med_abs_error": float(np.median(abs_temp_errors)) if abs_temp_errors else 0.0,
            }

        all_split_results[split_type] = results
        detailed_timing_records[split_type] = timing_list

    # Print & Save Comprehensive Report
    lines = [
        "# Iteration B: Selective Detector Recovery Benchmark Report",
        "**Evaluated Splits**: `split_in_distribution` (15 videos) vs `split_geo_aware` (15 videos) | **GT Tolerance**: $\\pm 35$ frames (~1.17s)",
        "",
        "## 1. Primary Comparative Incident Detection Metrics",
    ]

    for split_type in ["split_in_distribution", "split_geo_aware"]:
        lines.extend([
            f"### Split: `{split_type}`",
            "| Configuration | TP | FP (Ep) | FN | Precision | Recall | F1 Score | FP / Vid | Raw FP / Vid | Med Lead Time | Med Abs Error |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for cid, r in all_split_results[split_type].items():
            lines.append(
                f"| **{r['label']}** | {r['tp']} | {r['fp_episodes']} | {r['fn']} | "
                f"**{r['precision']:.3f}** | **{r['recall']:.3f}** | **{r['f1']:.3f}** | "
                f"{r['fp_per_vid']:.2f} | {r['raw_fp_per_vid']:.1f} | "
                f"+{r['med_lead_sec']:.2f}s ({r['med_lead_frames']:+.0f}f) | {r['med_abs_error']:.0f} frames |"
            )
        lines.append("")

    lines.extend([
        "## 2. Temporal False Positive Breakdown",
        "| Split | Configuration | Pre-Impact FP | Post-Impact FP | Repeated Incident FP | Total FP Episodes |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    for split_type, s_name in [("split_in_distribution", "In-Dist"), ("split_geo_aware", "Geo-Aware")]:
        for cid, r in all_split_results[split_type].items():
            lines.append(
                f"| **{s_name}** | **{r['label']}** | "
                f"{r['pre_impact_fp']} | {r['post_impact_fp']} | {r['repeated_fp']} | **{r['fp_episodes']}** |"
            )

    lines.extend([
        "",
        "## 3. Detailed Ground Truth vs. Predicted Timing Breakdown (Iteration B)",
        "| Split | Video ID | GT Annotation Frame | Predicted Trigger Frame | Temporal Error | Lead Time | Outcome |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for split_type, s_name in [("split_in_distribution", "In-Dist"), ("split_geo_aware", "Geo-Aware")]:
        for row in detailed_timing_records[split_type]:
            p_f = f"{row['pred_frame']}" if row['pred_frame'] is not None else "None"
            e_f = f"{row['temporal_error_frames']:+d}f" if row['temporal_error_frames'] is not None else "--"
            l_s = f"{row['lead_time_sec']:+.2f}s" if row['lead_time_sec'] is not None else "--"
            lines.append(
                f"| **{s_name}** | `{row['video_id']}` | {row['gt_frame']} | {p_f} | {e_f} | {l_s} | **{row['outcome']}** |"
            )

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")

    with open("docs/iteration_b_detector_recovery_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Report saved to: docs/iteration_b_detector_recovery_report.md")


if __name__ == "__main__":
    run_benchmark()
