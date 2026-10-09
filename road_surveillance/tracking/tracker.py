"""ByteTrack tracking module."""

import logging
from typing import Dict, List, Tuple
import numpy as np
import supervision as sv

from .track_state import Track, TrackletState
from ..detection.detector_interface import DetectionBatch

logger = logging.getLogger("road_surveillance.tracking.bytetrack")


class ByteTrackTracker:
    """ByteTrack Multi-Object Tracker integrating supervision.ByteTrack."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )
        # Ensure det_thresh allows track initialization down to track_activation_threshold
        self.tracker.det_thresh = float(track_activation_threshold)
        self.tracks: Dict[int, Track] = {}
        logger.info("Initialized ByteTrackTracker with det_thresh=%.2f", self.tracker.det_thresh)

    def reset(self) -> None:
        """Reset internal tracker state (e.g. between videos)."""
        self.tracker.reset()
        self.tracks.clear()

    def update(self, batch: DetectionBatch) -> Tuple[List[TrackletState], Dict[int, Track]]:
        """Update tracker with detection batch and return current tracklet states and tracks."""
        tracklet_states: List[TrackletState] = []

        VEHICLE_MAP = {
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "van": "van",
            "motorcycle": "motorcycle",
            "train": "truck",  # Remap roadway train detections to heavy vehicle/truck
        }

        if len(batch.boxes) == 0:
            # Handle empty frame with coasting for recently active mature vehicle tracks
            for tid, track in list(self.tracks.items()):
                is_veh = track.get_effective_label() in Track.VEHICLE_CLASSES
                can_coast = (
                    track.track_age >= 3
                    and is_veh
                    and track.coasting_count < 8
                    and track.status in ("active", "coasting")
                )
                if can_coast:
                    track.coast(batch.pts_ns)
                    gc_px = Track.calculate_ground_contact(track.current_bbox_xywh)
                    tracklet = TrackletState(
                        video_id=batch.video_id,
                        track_id=tid,
                        class_id=track.class_id,
                        class_label=track.get_effective_label(),
                        frame_idx=batch.frame_idx,
                        pts_ns=batch.pts_ns,
                        bbox_xywh=track.current_bbox_xywh,
                        ground_contact_px=gc_px,
                        track_status="coasting",
                    )
                    tracklet_states.append(tracklet)
                else:
                    track.mark_lost()
                    if track.status == "removed":
                        del self.tracks[tid]
            return tracklet_states, self.tracks

        # Prepare supervision Detections
        sv_detections = sv.Detections(
            xyxy=batch.boxes,
            confidence=batch.confidences,
            class_id=batch.class_ids,
        )

        tracked_detections = self.tracker.update_with_detections(sv_detections)

        active_tracker_ids = set()

        # Create class_id -> label mapping from current batch
        cid_to_label = {
            int(c): str(lbl)
            for c, lbl in zip(batch.class_ids, batch.class_labels)
        }

        if len(tracked_detections) > 0 and tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                tid = int(tracked_detections.tracker_id[i])
                active_tracker_ids.add(tid)

                box_xyxy = tracked_detections.xyxy[i]
                bx1, by1, bx2, by2 = box_xyxy
                bw = bx2 - bx1
                bh = by2 - by1
                bbox_xywh = (float(bx1), float(by1), float(bw), float(bh))

                cid = int(tracked_detections.class_id[i]) if tracked_detections.class_id is not None else 0
                conf = float(tracked_detections.confidence[i]) if tracked_detections.confidence is not None else 1.0
                raw_label = cid_to_label.get(cid, "object")
                label = VEHICLE_MAP.get(raw_label.lower(), raw_label)

                # Update or create Track
                if tid not in self.tracks:
                    self.tracks[tid] = Track(
                        track_id=tid,
                        class_id=cid,
                        class_label=label,
                    )

                track = self.tracks[tid]
                track.update(bbox_xywh=bbox_xywh, confidence=conf, pts_ns=batch.pts_ns, label=label)
                effective_label = track.get_effective_label()

                gc_px = Track.calculate_ground_contact(bbox_xywh)

                tracklet = TrackletState(
                    video_id=batch.video_id,
                    track_id=tid,
                    class_id=track.class_id,
                    class_label=effective_label,
                    frame_idx=batch.frame_idx,
                    pts_ns=batch.pts_ns,
                    bbox_xywh=bbox_xywh,
                    ground_contact_px=gc_px,
                    track_status="active",
                )
                tracklet_states.append(tracklet)

        # Mark non-updated tracks with deformation coasting
        for tid, track in list(self.tracks.items()):
            if tid not in active_tracker_ids:
                is_veh = track.get_effective_label() in Track.VEHICLE_CLASSES
                can_coast = (
                    track.track_age >= 3
                    and is_veh
                    and track.coasting_count < 8
                    and track.status in ("active", "coasting")
                )
                if can_coast:
                    track.coast(batch.pts_ns)
                    gc_px = Track.calculate_ground_contact(track.current_bbox_xywh)
                    tracklet = TrackletState(
                        video_id=batch.video_id,
                        track_id=tid,
                        class_id=track.class_id,
                        class_label=track.get_effective_label(),
                        frame_idx=batch.frame_idx,
                        pts_ns=batch.pts_ns,
                        bbox_xywh=track.current_bbox_xywh,
                        ground_contact_px=gc_px,
                        track_status="coasting",
                    )
                    tracklet_states.append(tracklet)
                else:
                    track.mark_lost()
                    if track.status == "removed":
                        del self.tracks[tid]

        return tracklet_states, self.tracks
