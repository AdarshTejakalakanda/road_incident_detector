"""Run complete Detector A/B Benchmark Preflight Suite.
Executes Phase 1-4 checks without modifying production code.
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.detector_ab_adapters import (
    MAPPING_VERSION,
    UNIFIED_TAXONOMY,
    UltralyticsYOLOAdapter,
    RTDETRv2Adapter,
    ResourceMonitor,
    StandardDetectionOutput,
)
from road_surveillance.tracking.tracker import ByteTrackTracker
from road_surveillance.detection.detector_interface import DetectionBatch


# ---------------------------------------------------------------------------
# Setup Unified Class Maps
# ---------------------------------------------------------------------------

MAP_A = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MAP_B = {
    0: "car",          # Hatchback
    1: "car",          # Sedan
    2: "car",          # SUV
    3: "van",          # MUV
    4: "bus",          # Bus
    5: "truck",        # Truck
    6: "auto_rickshaw",# Three-wheeler
    7: "motorcycle",   # Two-wheeler
    8: "truck",        # LCV
    9: "bus",          # Mini-bus
    10: "van",         # Tempo-traveller
    11: "bicycle",     # Bicycle
    12: "van",         # Van
    13: "other_vehicle"# Others
}

MAP_C = dict(MAP_B)

MAP_D = {
    0: "auto_rickshaw",# auto
    1: "bus",          # bus
    2: "car",          # car
    3: "truck",        # lcv
    4: "motorcycle",   # motorcycle
    5: "truck",        # multiaxle
    6: "other_vehicle",# tractor
    7: "truck",        # truck
}

MAP_E = {
    0: "car",          # car
    1: "truck",        # truck
    2: "bus",          # bus
}


def build_adapters():
    adapters = {
        "Arm A": UltralyticsYOLOAdapter(
            arm_id="Arm A",
            model_name="YOLO26n (Control)",
            weights_path="models/yolo26n.pt",
            unified_map=MAP_A,
        ),
        "Arm B": UltralyticsYOLOAdapter(
            arm_id="Arm B",
            model_name="UVH-26 YOLO11-S",
            weights_path="models/benchmark_arms/UVH-26-MV-YOLOv11-S.pt",
            unified_map=MAP_B,
        ),
        "Arm C": RTDETRv2Adapter(
            arm_id="Arm C",
            model_name="UVH-26 RT-DETRv2-S",
            config_path="models/benchmark_arms/rtdetr_s_flattened.yaml",
            weights_path="models/benchmark_arms/UVH-26-MV-RT-DETRv2-S.pth",
            unified_map=MAP_C,
        ),
        "Arm D": UltralyticsYOLOAdapter(
            arm_id="Arm D",
            model_name="vietnguyennn0705 Highway YOLOv8m",
            weights_path="models/benchmark_arms/vietnguyen_best.pt",
            unified_map=MAP_D,
        ),
        "Arm E": UltralyticsYOLOAdapter(
            arm_id="Arm E",
            model_name="G2mus Night-vehicle YOLOv8s",
            weights_path="models/benchmark_arms/night-vehicle-yolov8s.pt",
            unified_map=MAP_E,
        ),
    }
    return adapters


def run_contract_validation(adapters, test_frame):
    print("=== Phase 2: Adapter Contract Validation ===")
    results = {}
    empty_frame = np.zeros((720, 960, 3), dtype=np.uint8)

    for name, adapter in adapters.items():
        # 1. Coordinate convention & shape check
        out = adapter.detect(test_frame, frame_id="frame_000")
        assert out.frame_id == "frame_000", "Frame identity mismatch"
        assert out.mapping_version == MAPPING_VERSION, "Mapping version mismatch"
        assert len(out.boxes) == len(out.scores) == len(out.class_ids) == len(out.class_names) == len(out.unified_class_ids) == len(out.unified_class_names)

        coord_valid = True
        for b in out.boxes:
            x1, y1, x2, y2 = b
            if not (x1 <= x2 and y1 <= y2 and x1 >= 0 and y1 >= 0):
                coord_valid = False
                break

        # 2. Empty output handling
        empty_out = adapter.detect(empty_frame, frame_id="frame_empty", conf_threshold=0.99)
        empty_valid = (
            len(empty_out.boxes) == 0
            and empty_out.boxes.shape == (0, 4)
            and len(empty_out.scores) == 0
            and len(empty_out.class_ids) == 0
        )

        # 3. Class mapping round trip
        round_trip_valid = True
        for raw_id, u_name in zip(out.class_ids, out.unified_class_names):
            _, mapped_name = adapter.map_to_unified(int(raw_id))
            if mapped_name != u_name:
                round_trip_valid = False
                break

        res = {
            "coordinate_convention": "PASS (xyxy, non-negative)" if coord_valid else "FAIL",
            "empty_output_handling": "PASS" if empty_valid else "FAIL",
            "class_mapping_roundtrip": "PASS" if round_trip_valid else "FAIL",
            "frame_identity": "PASS" if out.frame_id == "frame_000" else "FAIL",
            "detections_count": len(out.boxes),
        }
        results[name] = res
        print(f"[{name}] {res}")

    return results


def run_determinism_test(adapters, test_frame):
    print("\n=== Phase 4: Determinism & Sanity Test ===")
    det_results = {}

    for name, adapter in adapters.items():
        t0 = time.perf_counter()
        out1 = adapter.detect(test_frame, frame_id="run_1")
        lat1 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        out2 = adapter.detect(test_frame, frame_id="run_2")
        lat2 = (time.perf_counter() - t0) * 1000.0

        # Output determinism: exact match on boxes and scores
        box_diff = np.max(np.abs(out1.boxes - out2.boxes)) if len(out1.boxes) > 0 else 0.0
        score_diff = np.max(np.abs(out1.scores - out2.scores)) if len(out1.scores) > 0 else 0.0
        class_match = np.array_equal(out1.class_ids, out2.class_ids)

        # Calculate IoU between identical runs
        if len(out1.boxes) > 0:
            # pairwise IoU for identical boxes
            b1 = out1.boxes
            b2 = out2.boxes
            inter_x1 = np.maximum(b1[:, 0], b2[:, 0])
            inter_y1 = np.maximum(b1[:, 1], b2[:, 1])
            inter_x2 = np.minimum(b1[:, 2], b2[:, 2])
            inter_y2 = np.minimum(b1[:, 3], b2[:, 3])
            inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
            b1_area = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
            b2_area = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
            iou = inter_area / (b1_area + b2_area - inter_area + 1e-7)
            min_iou = float(np.min(iou))
        else:
            min_iou = 1.0

        is_deterministic = (box_diff == 0.0 and score_diff == 0.0 and class_match)

        det_results[name] = {
            "deterministic_output": "PASS (Exact bitwise match)" if is_deterministic else "FAIL",
            "max_box_delta_px": float(box_diff),
            "max_score_delta": float(score_diff),
            "min_iou": float(min_iou),
            "pass1_latency_ms": round(lat1, 2),
            "pass2_latency_ms": round(lat2, 2),
            "timing_variation_ms": round(abs(lat1 - lat2), 2),
            "proposed_fp32_tolerance": "PROPOSED, NOT APPROVED (IoU >= 0.99, score Delta <= 0.001)",
        }
        print(f"[{name}] Determinism: {det_results[name]['deterministic_output']}, Min IoU: {min_iou:.4f}, Timing var: {abs(lat1 - lat2):.2f}ms")

    return det_results


def run_bytetrack_integration(adapters, video_path):
    print("\n=== Phase 2: ByteTrack Integration on Short Clip (10 frames) ===")
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(10):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    print(f"Extracted {len(frames)} frames for short clip test.")
    bt_results = {}

    for name, adapter in adapters.items():
        # Read-only copy of production tracker configuration:
        tracker = ByteTrackTracker(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )

        all_track_ids = set()
        total_tracklets = 0
        t0 = time.perf_counter()

        for idx, img in enumerate(frames):
            out = adapter.detect(img, frame_id=idx, conf_threshold=0.25)
            # Create DetectionBatch compatible with pipeline
            batch = DetectionBatch(
                video_id="5uh8-3q51uI_00_short",
                frame_idx=idx,
                pts_ns=int(idx * (1e9 / 30.0)),
                boxes=out.boxes,
                confidences=out.scores,
                class_ids=out.unified_class_ids,
                class_labels=out.unified_class_names,
            )
            tracklets, tracks = tracker.update(batch)
            total_tracklets += len(tracklets)
            all_track_ids.update(tracks.keys())

        elapsed = time.perf_counter() - t0
        avg_fps = len(frames) / elapsed if elapsed > 0 else 0.0

        bt_results[name] = {
            "status": "PASS",
            "frames_processed": len(frames),
            "total_tracklets_emitted": total_tracklets,
            "unique_tracks_initialized": len(all_track_ids),
            "total_time_sec": round(elapsed, 3),
            "offline_throughput_fps": round(avg_fps, 2),
            "errors": None,
        }
        print(f"[{name}] ByteTrack Integration: PASS, Unique Tracks: {len(all_track_ids)}, Tracklets: {total_tracklets}, Throughput: {avg_fps:.1f} FPS")

    return bt_results


def run_telemetry_traces(adapters, test_frame):
    print("\n=== Phase 3: Telemetry & Resource Monitor ===")
    monitor = ResourceMonitor()
    traces = {}

    for name, adapter in adapters.items():
        t0 = time.perf_counter()
        adapter.detect(test_frame, frame_id="telemetry_test")
        lat = (time.perf_counter() - t0) * 1000.0
        record = monitor.measure(latency_ms=lat)
        traces[name] = asdict(record)
        print(f"[{name}] Latency: {record.latency_ms}ms, CPU: {record.cpu_percent}%, RAM Used: {record.ram_used_mb}MB, Available: {record.ram_available_mb}MB")

    return traces


def build_night_manifest_template():
    print("\n=== Phase 4: Night Holdout Manifest Template ===")
    metadata_path = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\metadata-real.csv"
    with open(metadata_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0]
    data_rows = lines[1:]

    night_records = []
    for r in data_rows:
        parts = r.split(",")
        # day_time is column index 14
        if len(parts) > 14 and parts[14].lower() == "night":
            night_records.append(r)

    manifest_template = {
        "manifest_name": "night_holdout_template",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": metadata_path,
        "criteria": {
            "day_time": "night",
            "min_night_videos": 8,
            "min_cameras": 3,
            "min_night_instances": 300,
        },
        "statistics": {
            "total_night_videos_found": len(night_records),
            "distinct_regions": len(set(r.split(",")[11] for r in night_records)),
            "scene_layouts": list(set(r.split(",")[12] for r in night_records)),
            "quality_distribution": list(set(r.split(",")[15] for r in night_records)),
        },
        "sample_manifest_entries_top5": night_records[:5],
        "holdout_usage_state": "TEMPLATE_ONLY (HOLD-OUT UNUSED FOR TRAINING/CALIBRATION/INFERENCE)",
    }

    manifest_json = json.dumps(manifest_template, indent=2)
    manifest_path = "models/benchmark_arms/night_holdout_manifest_template.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_json)

    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    print(f"Saved manifest template to {manifest_path}")
    print(f"Manifest Template SHA-256: {manifest_hash}")
    return manifest_template, manifest_hash


def main():
    print("Starting Preflight Verification Suite...")
    test_frame_path = "models/benchmark_arms/test_frame.jpg"
    test_frame = cv2.imread(test_frame_path)
    video_path = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive\real_videos\5uh8-3q51uI_00.mp4"

    adapters = build_adapters()

    # 1. Contract validation
    contracts = run_contract_validation(adapters, test_frame)

    # 2. Determinism test
    determinism = run_determinism_test(adapters, test_frame)

    # 3. Telemetry traces
    telemetry = run_telemetry_traces(adapters, test_frame)

    # 4. ByteTrack integration
    bytetrack = run_bytetrack_integration(adapters, video_path)

    # 5. Night holdout manifest template
    manifest_data, manifest_hash = build_night_manifest_template()

    # Save full preflight audit JSON
    audit = {
        "preflight_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_version": MAPPING_VERSION,
        "unified_taxonomy": UNIFIED_TAXONOMY,
        "contracts": contracts,
        "determinism": determinism,
        "telemetry": telemetry,
        "bytetrack_integration": bytetrack,
        "night_manifest_hash": manifest_hash,
    }

    with open("models/benchmark_arms/preflight_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\nPreflight Audit successfully saved to models/benchmark_arms/preflight_audit_results.json")


if __name__ == "__main__":
    main()
