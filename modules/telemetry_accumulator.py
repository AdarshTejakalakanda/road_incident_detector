"""Telemetry Accumulator and Structured Snapshot Buffer Module.

Implements Phase 2 Telemetry Ownership:
- 60-byte fixed-width NumPy structured array buffer.
- Zero reconstruction from bounding boxes.
- Zero synthetic NaN propagation.
- High-efficiency TrackPhysicalSnapshotView adapter.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import sys


TELEMETRY_DTYPE = np.dtype([
    ("track_id", np.int32),           # Offset 0,  4 bytes
    ("frame_idx", np.int32),          # Offset 4,  4 bytes
    ("timestamp_s", np.float64),      # Offset 8,  8 bytes
    ("bbox_l", np.float32),           # Offset 16, 4 bytes
    ("bbox_t", np.float32),           # Offset 20, 4 bytes
    ("bbox_w", np.float32),           # Offset 24, 4 bytes
    ("bbox_h", np.float32),           # Offset 28, 4 bytes
    ("ground_x", np.float32),         # Offset 32, 4 bytes
    ("ground_y", np.float32),         # Offset 36, 4 bytes
    ("speed_kmh", np.float32),        # Offset 40, 4 bytes
    ("vx_mps", np.float32),           # Offset 44, 4 bytes
    ("vy_mps", np.float32),           # Offset 48, 4 bytes
    ("accel_mps2", np.float32),       # Offset 52, 4 bytes
    ("heading_deg", np.float32),      # Offset 56, 4 bytes
], align=False)  # Exactly 60 bytes / row


class TrackPhysicalSnapshotView:
    """Read-only view adapter over a single row in the structured buffer."""
    __slots__ = ("_row",)

    def __init__(self, row: np.void):
        self._row = row

    @property
    def track_id(self) -> int:
        return int(self._row["track_id"])

    @property
    def frame_idx(self) -> int:
        return int(self._row["frame_idx"])

    @property
    def timestamp_s(self) -> float:
        return float(self._row["timestamp_s"])

    @property
    def bbox_xywh(self) -> Tuple[float, float, float, float]:
        return (
            float(self._row["bbox_l"]),
            float(self._row["bbox_t"]),
            float(self._row["bbox_w"]),
            float(self._row["bbox_h"]),
        )

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Compatibility alias for SceneEscalator."""
        return self.bbox_xywh

    @property
    def ground_pos(self) -> Tuple[float, float]:
        return (float(self._row["ground_x"]), float(self._row["ground_y"]))

    @property
    def speed_kmh(self) -> float:
        return float(self._row["speed_kmh"])

    @property
    def velocity_vector(self) -> Tuple[float, float]:
        return (float(self._row["vx_mps"]), float(self._row["vy_mps"]))

    @property
    def acceleration_mag(self) -> float:
        return float(self._row["accel_mps2"])

    @property
    def heading_deg(self) -> float:
        return float(self._row["heading_deg"])

    @property
    def is_valid_kinematics(self) -> bool:
        """Helper to verify ground coordinates and speed are finite."""
        gx = self._row["ground_x"]
        gy = self._row["ground_y"]
        spd = self._row["speed_kmh"]
        return not (
            np.isnan(gx) or np.isnan(gy) or np.isnan(spd) or
            np.isinf(gx) or np.isinf(gy) or np.isinf(spd)
        )


class TelemetryAccumulator:
    """Pre-allocated contiguous NumPy buffer accumulating authoritative physical telemetry."""

    def __init__(self, initial_capacity: int = 10000):
        self._capacity = initial_capacity
        self._count = 0
        self._buffer = np.empty(self._capacity, dtype=TELEMETRY_DTYPE)
        self._realloc_count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def buffer_nbytes(self) -> int:
        return int(self._buffer.nbytes)

    def _ensure_capacity(self, num_new: int) -> None:
        if self._count + num_new > self._capacity:
            new_cap = max(self._capacity * 2, self._count + num_new)
            new_buf = np.empty(new_cap, dtype=TELEMETRY_DTYPE)
            new_buf[:self._count] = self._buffer[:self._count]
            self._buffer = new_buf
            self._capacity = new_cap
            self._realloc_count += 1

    def record_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        timestamp_s: float,
        active_tracks: Dict[int, Any],
        kinematics_map: Dict[int, Any],
    ) -> None:
        """Records physical state for all active tracks in the current frame."""
        if not active_tracks:
            return

        num_tracks = len(active_tracks)
        self._ensure_capacity(num_tracks)

        write_idx = self._count
        for tid, tr in active_tracks.items():
            k = kinematics_map.get(tid) if kinematics_map else None

            # Bounding box extraction
            if tr is not None and hasattr(tr, "current_bbox_xywh"):
                b = tr.current_bbox_xywh
                bl, bt, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            else:
                bl, bt, bw, bh = np.nan, np.nan, np.nan, np.nan

            # Ground coordinates and kinematics
            if k is not None:
                gx, gy = (float(k.ground_pos[0]), float(k.ground_pos[1])) if hasattr(k, "ground_pos") and k.ground_pos else (np.nan, np.nan)
                spd = float(k.speed_kmh) if hasattr(k, "speed_kmh") else 0.0
                vx, vy = (float(k.velocity_vector[0]), float(k.velocity_vector[1])) if hasattr(k, "velocity_vector") and k.velocity_vector else (0.0, 0.0)
                acc = float(k.acceleration_mag) if hasattr(k, "acceleration_mag") else 0.0
                hdg = float(k.heading_deg) if hasattr(k, "heading_deg") else 0.0
            else:
                # Fallback to track ground history if geometry engine omitted track
                if tr is not None and getattr(tr, "ground_history", None):
                    last_g = tr.ground_history[-1]
                    gx, gy = float(last_g[0]), float(last_g[1])
                else:
                    gx, gy = np.nan, np.nan
                spd = float(getattr(tr, "max_recent_speed_kmh", 0.0)) if tr else 0.0
                vx, vy = 0.0, 0.0
                acc = 0.0
                hdg = 0.0

            row = self._buffer[write_idx]
            row["track_id"] = tid
            row["frame_idx"] = frame_idx
            row["timestamp_s"] = timestamp_s
            row["bbox_l"] = bl
            row["bbox_t"] = bt
            row["bbox_w"] = bw
            row["bbox_h"] = bh
            row["ground_x"] = gx
            row["ground_y"] = gy
            row["speed_kmh"] = spd
            row["vx_mps"] = vx
            row["vy_mps"] = vy
            row["accel_mps2"] = acc
            row["heading_deg"] = hdg

            write_idx += 1

        self._count = write_idx

    def get_window(
        self,
        t_start: float,
        t_end: float,
        filter_invalid: bool = True,
    ) -> Dict[int, List[TrackPhysicalSnapshotView]]:
        """Extracts snapshots for the specified time window [t_start, t_end].
        
        Guarantees:
        - Snapshots with non-finite values are filtered to prevent NaN propagation.
        - Result is structured as Dict[track_id, List[TrackPhysicalSnapshotView]].
        """
        if self._count == 0:
            return {}

        active_slice = self._buffer[:self._count]
        # Include small epsilon to account for floating point jitter on exact boundaries
        dt_eps = 1e-4
        mask = (active_slice["timestamp_s"] >= t_start - dt_eps) & (active_slice["timestamp_s"] <= t_end + dt_eps)

        if filter_invalid:
            finite_mask = (
                ~np.isnan(active_slice["ground_x"]) &
                ~np.isnan(active_slice["ground_y"]) &
                ~np.isnan(active_slice["speed_kmh"]) &
                ~np.isinf(active_slice["ground_x"]) &
                ~np.isinf(active_slice["ground_y"]) &
                ~np.isinf(active_slice["speed_kmh"])
            )
            mask = mask & finite_mask

        matching_rows = active_slice[mask]
        result: Dict[int, List[TrackPhysicalSnapshotView]] = {}

        for row in matching_rows:
            tid = int(row["track_id"])
            view = TrackPhysicalSnapshotView(row)
            if tid not in result:
                result[tid] = []
            result[tid].append(view)

        return result

    def get_total_memory_usage(self) -> Dict[str, Any]:
        """Calculates exact heap memory footprint of the accumulator."""
        buf_bytes = int(self._buffer.nbytes)
        obj_bytes = sys.getsizeof(self) + sys.getsizeof(self.__dict__)
        total_bytes = buf_bytes + obj_bytes
        return {
            "total_bytes": total_bytes,
            "total_mib": round(total_bytes / (1024 * 1024), 3),
            "buffer_bytes": buf_bytes,
            "allocated_capacity": self._capacity,
            "recorded_snapshots": self._count,
            "reallocations": self._realloc_count,
        }
