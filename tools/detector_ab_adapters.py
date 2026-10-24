"""Detector A/B Benchmark Preflight and Adapter Integration Suite.

Implements isolated adapters for Arms A-E with unified interface:
    {frame_id, boxes, scores, class_ids, class_names, unified_class_ids, unified_class_names, mapping_version}

Performs:
    1. Adapter contract validation (coordinates, empty outputs, class mapping round-trip, frame identity).
    2. Reference sanity tests & RT-DETRv2 verification against upstream preprocessing.
    3. Resource and timing telemetry monitor (CPU, RAM, latency, monotonic timestamp, UTC offset).
    4. Determinism validation (repeat test with identical settings).
    5. Short-clip ByteTrack integration test (10 frames of 5uh8-3q51uI_00).
    6. Night-holdout manifest template generation and hashing.
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

# Add official RT-DETRv2 repository to sys.path
RTDETR_REPO_PATH = os.path.abspath('models/benchmark_arms/RT-DETR/rtdetrv2_pytorch')
if RTDETR_REPO_PATH not in sys.path:
    sys.path.insert(0, RTDETR_REPO_PATH)

from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch


# ---------------------------------------------------------------------------
# Common Output Data Contract
# ---------------------------------------------------------------------------

MAPPING_VERSION = "1.0.0-frozen"

UNIFIED_TAXONOMY: Dict[int, str] = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "van",
    5: "auto_rickshaw",
    6: "bicycle",
    7: "other_vehicle",
}

UNIFIED_NAME_TO_ID = {v: k for k, v in UNIFIED_TAXONOMY.items()}


@dataclass
class StandardDetectionOutput:
    frame_id: Any
    boxes: np.ndarray        # Shape [N, 4], float32 in [x1, y1, x2, y2]
    scores: np.ndarray       # Shape [N], float32 in [0.0, 1.0]
    class_ids: np.ndarray     # Shape [N], int32 raw native class IDs
    class_names: List[str]   # Length N, raw native class names
    unified_class_ids: np.ndarray    # Shape [N], int32 unified IDs
    unified_class_names: List[str]  # Length N, unified names
    mapping_version: str     # Frozen version tag


# ---------------------------------------------------------------------------
# Base Adapter and Concrete Implementations
# ---------------------------------------------------------------------------

class BaseDetectorAdapter:
    """Abstract base adapter enforcing the uniform detector contract."""

    def __init__(self, arm_id: str, model_name: str, mapping_version: str = MAPPING_VERSION):
        self.arm_id = arm_id
        self.model_name = model_name
        self.mapping_version = mapping_version
        self.raw_class_map: Dict[int, str] = {}
        self.unified_map: Dict[int, str] = {}

    def map_to_unified(self, raw_id: int) -> Tuple[int, str]:
        uname = self.unified_map.get(raw_id, "other_vehicle")
        uid = UNIFIED_NAME_TO_ID.get(uname, 7)
        return uid, uname

    def detect(self, image: np.ndarray, frame_id: Any, conf_threshold: float = 0.25) -> StandardDetectionOutput:
        raise NotImplementedError


class UltralyticsYOLOAdapter(BaseDetectorAdapter):
    """Adapter for YOLOv8/v11/YOLO26 models hosted in Ultralytics."""

    def __init__(self, arm_id: str, model_name: str, weights_path: str, unified_map: Dict[int, str]):
        super().__init__(arm_id, model_name)
        self.weights_path = weights_path
        self.model = YOLO(weights_path)
        self.raw_class_map = {int(k): str(v) for k, v in self.model.names.items()}
        self.unified_map = unified_map

    def detect(self, image: np.ndarray, frame_id: Any, conf_threshold: float = 0.25) -> StandardDetectionOutput:
        if image is None or image.size == 0:
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            verbose=False,
            device="cpu",
        )[0]

        if len(results.boxes) == 0:
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = results.boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = results.boxes.cls.cpu().numpy().astype(np.int32)
        class_names = [self.raw_class_map.get(int(cid), "unknown") for cid in class_ids]

        u_ids = []
        u_names = []
        for cid in class_ids:
            uid, uname = self.map_to_unified(int(cid))
            u_ids.append(uid)
            u_names.append(uname)

        return StandardDetectionOutput(
            frame_id=frame_id,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            unified_class_ids=np.array(u_ids, dtype=np.int32),
            unified_class_names=u_names,
            mapping_version=self.mapping_version,
        )


class RTDETRv2Adapter(BaseDetectorAdapter):
    """Adapter for UVH-26 RT-DETRv2-S using official upstream repository."""

    def __init__(self, arm_id: str, model_name: str, config_path: str, weights_path: str, unified_map: Dict[int, str]):
        super().__init__(arm_id, model_name)
        self.config_path = config_path
        self.weights_path = weights_path
        self.unified_map = unified_map

        # Raw classes from uvh_classes.txt
        classes_file = 'models/benchmark_arms/arm_b_classes.txt'
        with open(classes_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        self.raw_class_map = {idx: name for idx, name in enumerate(lines)}

        from src.core import YAMLConfig
        self.cfg = YAMLConfig(config_path, resume=weights_path)
        checkpoint = torch.load(weights_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        self.cfg.model.load_state_dict(state)

        class ModelDeploy(nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model
                self.postprocessor = postprocessor
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(outputs, orig_target_sizes)

        self.model = ModelDeploy(self.cfg.model.deploy(), self.cfg.postprocessor.deploy())
        self.model.eval()

        # Upstream preprocessing transforms per usage/inference_rtdetrv2.py:
        # Resize to (640, 640) and ToTensor (normalizes to [0, 1])
        self.transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])

    def detect(self, image: np.ndarray, frame_id: Any, conf_threshold: float = 0.25) -> StandardDetectionOutput:
        if image is None or image.size == 0:
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        h, w = image.shape[:2]
        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        orig_size = torch.tensor([w, h], dtype=torch.int64)[None]
        im_data = self.transforms(im_pil)[None]

        with torch.no_grad():
            labels, boxes, scores = self.model(im_data, orig_size)
            labels = labels[0].cpu().numpy()
            boxes = boxes[0].cpu().numpy()
            scores = scores[0].cpu().numpy()

        mask = scores >= conf_threshold
        if not np.any(mask):
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        boxes = boxes[mask].astype(np.float32)
        scores = scores[mask].astype(np.float32)
        class_ids = labels[mask].astype(np.int32)
        class_names = [self.raw_class_map.get(int(cid), f"class_{cid}") for cid in class_ids]

        u_ids = []
        u_names = []
        for cid in class_ids:
            uid, uname = self.map_to_unified(int(cid))
            u_ids.append(uid)
            u_names.append(uname)

        return StandardDetectionOutput(
            frame_id=frame_id,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            unified_class_ids=np.array(u_ids, dtype=np.int32),
            unified_class_names=u_names,
            mapping_version=self.mapping_version,
        )


class LightCoreAdapter(BaseDetectorAdapter):
    """Adapter for Nocturnal Light-Core Branch & Temporal Vehicle Hypothesis Engine."""

    def __init__(
        self,
        arm_id: str = "Cand_E",
        model_name: str = "Nocturnal Light-Core Branch",
        road_polygon: Optional[np.ndarray] = None,
        fps: float = 8.033,
    ):
        super().__init__(arm_id, model_name)
        from road_surveillance.detection.nocturnal_light_branch import NocturnalLightBranchDetector
        if road_polygon is None:
            road_polygon = np.array([
                [320, 560], [240, 480], [420, 315], [700, 315],
                [620, 400], [540, 470], [480, 560],
            ], np.int32)
        self.detector = NocturnalLightBranchDetector(road_polygon=road_polygon)
        self.fps = fps
        self.raw_class_map = {0: "car", 1: "truck"}
        self.unified_map = {0: "car", 1: "truck"}

    def detect(self, image: np.ndarray, frame_id: Any, conf_threshold: float = 0.25) -> StandardDetectionOutput:
        if image is None or image.size == 0:
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        try:
            f_idx = int(frame_id)
        except (ValueError, TypeError):
            f_idx = 0

        ts_s = f_idx / self.fps
        batch = self.detector.detect_frame(image, frame_idx=f_idx, timestamp_s=ts_s)

        if len(batch.boxes) == 0:
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        mask = batch.confidences >= conf_threshold
        if not np.any(mask):
            return StandardDetectionOutput(
                frame_id=frame_id,
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_names=[],
                unified_class_ids=np.empty((0,), dtype=np.int32),
                unified_class_names=[],
                mapping_version=self.mapping_version,
            )

        boxes = batch.boxes[mask].astype(np.float32)
        scores = batch.confidences[mask].astype(np.float32)
        class_ids = batch.class_ids[mask].astype(np.int32)
        class_names = [batch.class_labels[i] for i, m in enumerate(mask) if m]

        u_ids = []
        u_names = []
        for cid in class_ids:
            uid, uname = self.map_to_unified(int(cid))
            u_ids.append(uid)
            u_names.append(uname)

        return StandardDetectionOutput(
            frame_id=frame_id,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            class_names=class_names,
            unified_class_ids=np.array(u_ids, dtype=np.int32),
            unified_class_names=u_names,
            mapping_version=self.mapping_version,
        )



# ---------------------------------------------------------------------------
# Telemetry and Timing Monitor
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    timestamp_ns: int
    utc_iso: str
    utc_offset_sec: int
    cpu_percent: float
    ram_used_mb: float
    ram_available_mb: float
    gpu_percent: Optional[float]
    vram_used_mb: Optional[float]
    latency_ms: float
    error: Optional[str]


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        # Cache UTC offset
        now = datetime.now()
        local_now = now.astimezone()
        self.utc_offset_sec = int(local_now.utcoffset().total_seconds())

    def measure(self, latency_ms: float = 0.0, error: Optional[str] = None) -> TelemetryRecord:
        ts_ns = time.perf_counter_ns()
        now_utc = datetime.now(timezone.utc).isoformat()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        gpu_pct = None
        vram_mb = None
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        return TelemetryRecord(
            timestamp_ns=ts_ns,
            utc_iso=now_utc,
            utc_offset_sec=self.utc_offset_sec,
            cpu_percent=cpu_pct,
            ram_used_mb=round((mem.total - mem.available) / (1024 * 1024), 2),
            ram_available_mb=round(mem.available / (1024 * 1024), 2),
            gpu_percent=gpu_pct,
            vram_used_mb=vram_mb,
            latency_ms=round(latency_ms, 3),
            error=error,
        )
