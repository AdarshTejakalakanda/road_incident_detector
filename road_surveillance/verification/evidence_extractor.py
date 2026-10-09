"""Universal Shared Candidate Evidence Extractor for Multimodal Verification.

Supports all incident categories:
- Collision
- Wrong-Way
- Debris
- Animal
- Stalled Vehicle

Generates:
1. Full Context Keyframes (t-2 .. t+2)
2. Spatial ROI Crops (1.8x - 2.5x Zoom resized to ViT patch dimensions)
3. Set-of-Mark (SoM) Visual Prompts (Track badges and bounding indicators)
4. Dense Temporal ROI Sequence
5. Structured Kinematic Telemetry
"""

import os
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .base import CandidateEvidencePackage

logger = logging.getLogger("road_surveillance.verification.evidence_extractor")

# Distinct Set-of-Mark color palette (BGR)
SOM_COLORS = [
    (0, 0, 255),      # Bright Red (Track A)
    (255, 255, 0),    # Cyan (Track B)
    (0, 255, 255),    # Yellow (Track C)
    (0, 255, 0),      # Bright Green (Track D)
    (255, 0, 255),    # Magenta
]


class CandidateEvidenceExtractor:
    """Extracts synchronized visual keyframes, spatial ROI crops, and telemetry for candidates."""

    def __init__(
        self,
        video_dataset_dir: str = r"C:\Users\KALAKA~1\Downloads\archive\real_videos",
        pre_offset_frames: int = 15,
        post_offset_frames: int = 15,
        roi_target_size: Tuple[int, int] = (384, 384),
        roi_expansion_factor: float = 2.0,
    ):
        self.video_dataset_dir = os.path.abspath(video_dataset_dir)
        self.pre_offset_frames = pre_offset_frames
        self.post_offset_frames = post_offset_frames
        self.roi_target_size = roi_target_size
        self.roi_expansion_factor = roi_expansion_factor

    def get_video_path(self, video_id: str) -> Optional[str]:
        """Resolve the video file path from video_id."""
        for ext in [".mp4", ".avi", ".mov"]:
            candidate = os.path.join(self.video_dataset_dir, f"{video_id}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    def extract_keyframes_from_video(
        self,
        video_path: str,
        target_frames: List[int],
    ) -> Dict[int, np.ndarray]:
        """Efficiently seek and extract specific target frames from a video file."""
        frames_dict = {}
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found: {video_path}")
            return frames_dict

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {video_path}")
            return frames_dict

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sorted_targets = sorted(list(set(target_frames)))

        for f_idx in sorted_targets:
            clamped_idx = max(0, min(f_idx, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_dict[f_idx] = frame

        cap.release()
        return frames_dict

    def compute_incident_roi(
        self,
        frame_shape: Tuple[int, int],
        track_boxes: Optional[Dict[int, Tuple[float, float, float, float]]],
        involved_track_ids: List[int],
        default_center: Optional[Tuple[float, float]] = None,
    ) -> Tuple[int, int, int, int]:
        """Compute the spatial bounding union ROI for all involved tracks with expansion margin.
        
        Returns:
            (x, y, w, h) clamped within frame bounds.
        """
        fh, fw = frame_shape[:2]
        boxes_to_merge = []

        if track_boxes:
            for tid in involved_track_ids:
                if tid in track_boxes:
                    bx, by, bw, bh = track_boxes[tid]
                    boxes_to_merge.append([bx, by, bx + bw, by + bh])

        if not boxes_to_merge:
            if default_center:
                cx, cy = default_center
                half_w = fw * 0.2
                half_h = fh * 0.2
                return (
                    int(max(0, cx - half_w)),
                    int(max(0, cy - half_h)),
                    int(min(fw, half_w * 2)),
                    int(min(fh, half_h * 2)),
                )
            # Default to full frame if no bounding boxes are known
            return (0, 0, fw, fh)

        boxes_arr = np.array(boxes_to_merge, dtype=np.float32)
        min_x = np.min(boxes_arr[:, 0])
        min_y = np.min(boxes_arr[:, 1])
        max_x = np.max(boxes_arr[:, 2])
        max_y = np.max(boxes_arr[:, 3])

        union_w = max_x - min_x
        union_h = max_y - min_y
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        # Expand box by roi_expansion_factor and make square to prevent aspect distortion
        side = max(union_w, union_h, 48.0) * self.roi_expansion_factor
        half_side = side / 2.0

        roi_x1 = max(0.0, center_x - half_side)
        roi_y1 = max(0.0, center_y - half_side)
        roi_x2 = min(float(fw), center_x + half_side)
        roi_y2 = min(float(fh), center_y + half_side)

        rx = int(roi_x1)
        ry = int(roi_y1)
        rw = int(max(1.0, roi_x2 - roi_x1))
        rh = int(max(1.0, roi_y2 - roi_y1))
        return (rx, ry, rw, rh)

    def extract_roi_crop(
        self,
        image: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop and resize region of interest to ViT target resolution."""
        rx, ry, rw, rh = roi_bbox
        crop = image[ry : ry + rh, rx : rx + rw]
        if crop.size == 0:
            return np.zeros((self.roi_target_size[1], self.roi_target_size[0], 3), dtype=np.uint8)
        return cv2.resize(crop, self.roi_target_size, interpolation=cv2.INTER_LINEAR)

    def annotate_som(
        self,
        full_image: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        track_boxes: Optional[Dict[int, Tuple[float, float, float, float]]],
        involved_track_ids: List[int],
    ) -> np.ndarray:
        """Generate Set-of-Mark (SoM) visual prompt on the cropped ROI."""
        annotated_full = full_image.copy()
        if track_boxes:
            for idx, tid in enumerate(involved_track_ids):
                if tid in track_boxes:
                    bx, by, bw, bh = [int(v) for v in track_boxes[tid]]
                    color = SOM_COLORS[idx % len(SOM_COLORS)]

                    # Bounding box
                    cv2.rectangle(annotated_full, (bx, by), (bx + bw, by + bh), color, 3)

                    # Label badge with solid background
                    label = f"Track {tid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    badge_y1 = max(0, by - th - 10)
                    badge_y2 = max(th + 10, by)
                    cv2.rectangle(annotated_full, (bx, badge_y1), (bx + tw + 10, badge_y2), color, -1)
                    cv2.putText(
                        annotated_full,
                        label,
                        (bx + 5, badge_y2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )

        # Crop the annotated ROI
        return self.extract_roi_crop(annotated_full, roi_bbox)

    def extract_evidence_package(
        self,
        video_id: str,
        candidate_frame: int,
        candidate_detector: str,
        involved_track_ids: List[int],
        telemetry: Optional[Dict[str, Any]] = None,
        track_boxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        cached_frames: Optional[Dict[int, np.ndarray]] = None,
    ) -> CandidateEvidencePackage:
        """Extract keyframes, ROI crops, and telemetry into a CandidateEvidencePackage.
        
        Extracts 5 dense temporal points:
        - t-2: Pre-approach window (frame - 2 * pre_offset)
        - t-1: Approach window (frame - pre_offset)
        - t0:  Incident conflict point (frame)
        - t+1: Immediate aftermath (frame + post_offset)
        - t+2: Resolution window (frame + 2 * post_offset)
        """
        f_tminus2 = max(0, candidate_frame - 2 * self.pre_offset_frames)
        f_tminus1 = max(0, candidate_frame - self.pre_offset_frames)
        f_t0 = candidate_frame
        f_tplus1 = candidate_frame + self.post_offset_frames
        f_tplus2 = candidate_frame + 2 * self.post_offset_frames

        temporal_keys = ["t-2", "t-1", "t0", "t+1", "t+2"]
        temporal_indices = {
            "t-2": f_tminus2,
            "t-1": f_tminus1,
            "t0": f_t0,
            "t+1": f_tplus1,
            "t+2": f_tplus2,
        }

        target_frame_numbers = list(temporal_indices.values())
        raw_frames = {}

        if cached_frames is not None:
            for f in target_frame_numbers:
                if f in cached_frames:
                    raw_frames[f] = cached_frames[f].copy()

        # If not all target frames were in cache, read from video file
        missing_frames = [f for f in target_frame_numbers if f not in raw_frames]
        if missing_frames:
            vid_path = self.get_video_path(video_id)
            if vid_path:
                disk_frames = self.extract_keyframes_from_video(vid_path, missing_frames)
                raw_frames.update(disk_frames)

        default_shape = (720, 1280, 3)

        # Assemble full frames
        full_frames: Dict[str, np.ndarray] = {}
        for k in temporal_keys:
            f_num = temporal_indices[k]
            full_frames[k] = raw_frames.get(f_num, np.zeros(default_shape, dtype=np.uint8))

        # Compute spatial ROI
        roi_bbox = self.compute_incident_roi(
            frame_shape=full_frames["t0"].shape,
            track_boxes=track_boxes,
            involved_track_ids=involved_track_ids,
        )

        # Generate ROI crops and Set-of-Mark crops
        roi_crops: Dict[str, np.ndarray] = {}
        som_crops: Dict[str, np.ndarray] = {}
        for k in temporal_keys:
            img = full_frames[k]
            roi_crops[k] = self.extract_roi_crop(img, roi_bbox)
            som_crops[k] = self.annotate_som(img, roi_bbox, track_boxes, involved_track_ids)

        # Annotate trigger frame for legacy keyframe representation
        legacy_trigger = full_frames["t0"].copy()
        if track_boxes:
            for tid, bbox in track_boxes.items():
                if tid in involved_track_ids:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(legacy_trigger, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(
                        legacy_trigger,
                        f"Track {tid}",
                        (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

        event_id = f"{video_id}_{candidate_detector}_{candidate_frame}"
        keyframes = {
            "pre": full_frames["t-1"],
            "trigger": legacy_trigger,
            "post": full_frames["t+1"],
        }
        keyframe_indices = {
            "pre": f_tminus1,
            "trigger": f_t0,
            "post": f_tplus1,
        }

        return CandidateEvidencePackage(
            event_id=event_id,
            video_id=video_id,
            candidate_frame=candidate_frame,
            candidate_detector=candidate_detector,
            involved_track_ids=involved_track_ids,
            keyframes=keyframes,
            keyframe_indices=keyframe_indices,
            full_frames=full_frames,
            roi_crops=roi_crops,
            som_crops=som_crops,
            roi_bbox=roi_bbox,
            temporal_indices=temporal_indices,
            telemetry_summary=telemetry or {},
            metadata={
                "pre_offset_frames": self.pre_offset_frames,
                "post_offset_frames": self.post_offset_frames,
                "roi_target_size": list(self.roi_target_size),
                "roi_expansion_factor": self.roi_expansion_factor,
            },
        )

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
        """Encode an OpenCV BGR image to a base64 JPEG string."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_params)
        if not success:
            raise ValueError("Failed to encode image to JPEG")
        return base64.b64encode(buffer).decode("utf-8")
