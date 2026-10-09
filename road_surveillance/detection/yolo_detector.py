"""YOLO26 Detection Module."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from ultralytics import YOLO

from .detector_interface import DetectionBatch
from ..video.ingestion import IngestedFrame

logger = logging.getLogger("road_surveillance.detection.yolo26")


class YOLO26Detector:
    """YOLO26 Object Detector wrapper with Adaptive Nocturnal Sensitivity."""

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "van"}
    HEADLIGHT_CLASSES = {"traffic light", "airplane"}

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        conf_threshold: float = 0.25,
        nocturnal_conf_threshold: float = 0.10,
        enable_nocturnal_headlight_remap: bool = True,
        input_resolution: int = 640,
        device: str = "cpu",
        frame_sampling_rate: int = 1,
        enable_adaptive_contrast: bool = False,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nocturnal_conf_threshold = nocturnal_conf_threshold
        self.enable_nocturnal_headlight_remap = enable_nocturnal_headlight_remap
        self.input_resolution = input_resolution
        self.device = device
        self.frame_sampling_rate = max(1, frame_sampling_rate)
        self.enable_adaptive_contrast = enable_adaptive_contrast

        logger.info(f"Loading YOLO26 model from '{self.model_path}' on device '{self.device}'...")
        self.model = YOLO(self.model_path)
        self.names: Dict[int, str] = self.model.names if hasattr(self.model, "names") else {}

        # Mandatory startup vocabulary logging per specification
        logger.info(f"YOLO26 loaded successfully with {len(self.names)} classes in vocabulary:")
        for cid, cname in sorted(self.names.items()):
            logger.info(f"  Class {cid}: {cname}")

    def detect(
        self,
        frame: IngestedFrame,
        is_low_light: bool = False,
        road_surface: Optional[Any] = None,
    ) -> DetectionBatch:
        """Run object detection on an IngestedFrame.
        
        Args:
            frame: IngestedFrame with image array.
            is_low_light: Whether frame luminance is below nocturnal threshold (mean_Y < 50).
            road_surface: Optional RoadSurfaceDetector for horizon and roadway gating.
        """
        # Handle frame sampling rate
        if frame.frame_idx % self.frame_sampling_rate != 0:
            return DetectionBatch(
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                pts_ns=frame.pts_ns,
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        img = frame.image
        # Optional legacy contrast enhancement if explicitly enabled
        if self.enable_adaptive_contrast and img is not None and len(img.shape) == 3:
            mean_b = float(np.mean(img))
            if mean_b < 55.0:
                import cv2
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                l_ch = clahe.apply(l_ch)
                img = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

        # Dynamic confidence threshold: lower only in low light for recovery
        active_conf = min(self.conf_threshold, self.nocturnal_conf_threshold) if is_low_light else self.conf_threshold

        # Run inference
        results = self.model.predict(
            source=img,
            conf=active_conf,
            imgsz=self.input_resolution,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionBatch(
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                pts_ns=frame.pts_ns,
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
        raw_labels = [self.names.get(int(cid), f"class_{cid}") for cid in class_ids]

        h_img, w_img = img.shape[:2]
        horizon_y = road_surface.get_horizon_y(h_img) if road_surface is not None else 0.0

        filtered_boxes = []
        filtered_confs = []
        filtered_cids = []
        filtered_labels = []

        for b_box, b_conf, b_cid, b_lbl in zip(boxes_xyxy, confidences, class_ids, raw_labels):
            lbl_lower = b_lbl.lower()
            if lbl_lower == "train":
                lbl_lower = "truck"

            conf_val = float(b_conf)
            bx1, by1, bx2, by2 = b_box
            gc_y = by2

            # Off-road / sky horizon gating
            if road_surface is not None and gc_y < horizon_y:
                continue

            if is_low_light:
                # Class-selective filtering in nocturnal mode
                if lbl_lower in self.VEHICLE_CLASSES:
                    if conf_val >= self.nocturnal_conf_threshold:
                        filtered_boxes.append(b_box)
                        filtered_confs.append(conf_val)
                        filtered_cids.append(b_cid)
                        filtered_labels.append(lbl_lower)
                elif self.enable_nocturnal_headlight_remap and lbl_lower in self.HEADLIGHT_CLASSES:
                    if conf_val >= self.nocturnal_conf_threshold:
                        filtered_boxes.append(b_box)
                        filtered_confs.append(conf_val)
                        filtered_cids.append(b_cid)
                        filtered_labels.append("car")
                else:
                    # Non-vehicle classes (pedestrian, signs) retain standard strict threshold
                    if conf_val >= self.conf_threshold:
                        filtered_boxes.append(b_box)
                        filtered_confs.append(conf_val)
                        filtered_cids.append(b_cid)
                        filtered_labels.append(lbl_lower)
            else:
                # Daytime mode: standard strict threshold
                if conf_val >= self.conf_threshold:
                    filtered_boxes.append(b_box)
                    filtered_confs.append(conf_val)
                    filtered_cids.append(b_cid)
                    filtered_labels.append(lbl_lower)

        if not filtered_boxes:
            return DetectionBatch(
                video_id=frame.video_id,
                frame_idx=frame.frame_idx,
                pts_ns=frame.pts_ns,
                boxes=np.empty((0, 4), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
            )

        final_boxes = np.array(filtered_boxes, dtype=np.float32)
        final_confs = np.array(filtered_confs, dtype=np.float32)
        final_cids = np.array(filtered_cids, dtype=np.int32)

        return DetectionBatch(
            video_id=frame.video_id,
            frame_idx=frame.frame_idx,
            pts_ns=frame.pts_ns,
            boxes=final_boxes,
            confidences=final_confs,
            class_ids=final_cids,
            class_labels=filtered_labels,
        )
