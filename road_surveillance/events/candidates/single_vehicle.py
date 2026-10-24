"""Single-Vehicle Incident Candidate Detector (Iteration C).

Detects single-vehicle anomalies independently from pairwise collisions:
- Severe barrier / roadway impact deceleration: exceeding tire-road friction limit
- Abnormal yaw / sudden trajectory spin-out: rapid angular heading rotation
- Rollover / geometric aspect ratio inversion: abrupt bounding box deformation
- Rapid deceleration with sudden trajectory arrest
Integrates track maturity gating, kinematic warm-up, and physical feasibility bounds.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

from ..base import CandidateDetector, EventCandidate
from ...tracking.track_state import Track
from ...geometry.spatial_metrics import TrackKinematics, TrajectoryConvergence
from ...calibration.lane_manager import LaneManager
from modules.single_vehicle import compute_single_vehicle_evidence, check_has_lead_vehicle


def is_physically_plausible_arrest(speed_kmh: float, accel_mps2: float) -> bool:
    """Evaluates whether longitudinal deceleration represents a physically plausible
    momentum arrest versus coordinate/Kalman jitter on a near-stationary vehicle.
    """
    if speed_kmh <= 0.1:
        return False
    speed_mps = speed_kmh / 3.6
    tau_rest = speed_mps / (abs(accel_mps2) + 1e-6)
    # Arrest requiring < 0.25s is physically implausible as sustained vehicle deceleration
    return tau_rest >= 0.25


DYNAMIC_ANOMALY_TYPES = {"yaw", "rollover", "spin_out", "composite"}


class SingleVehicleAnomalyDetector(CandidateDetector):
    """Detects single-vehicle incidents including barrier collisions, spin-outs, and rollovers."""

    VEHICLE_LABELS = {"car", "truck", "bus", "van", "motorcycle", "train", "vehicle", "unconfirmed"}

    def __init__(
        self,
        is_camera_calibrated: bool = False,
        video_warmup_frames: int = 15,
        min_track_age: int = 8,
        deceleration_threshold_ms2: float = -4.0,
        severe_impact_decel_ms2: float = -14.0,
        max_physical_decel_ms2: float = 60.0,
        yaw_rate_threshold_deg: float = 30.0,
        aspect_ratio_change_threshold: float = 0.35,
        sudden_stop_speed_drop_kmh: float = 15.0,
        edge_margin_px: float = 30.0,
        min_persistence_frames: int = 2,
        tau_single: float = 0.50,
        roadway_polygon: Optional[List[Tuple[float, float]]] = None,
    ):
        self.is_camera_calibrated = is_camera_calibrated
        self.video_warmup_frames = video_warmup_frames
        self.min_track_age = min_track_age
        self.decel_threshold = deceleration_threshold_ms2
        self.severe_impact_decel = severe_impact_decel_ms2
        self.max_physical_decel = max_physical_decel_ms2
        self.yaw_rate_thresh = yaw_rate_threshold_deg
        self.ar_change_thresh = aspect_ratio_change_threshold
        self.sudden_stop_drop = sudden_stop_speed_drop_kmh
        self.edge_margin_px = edge_margin_px
        self.min_persistence_frames = min_persistence_frames
        self.tau_single = tau_single
        self.roadway_polygon = roadway_polygon

        # Track history: track_id -> list of feature dicts
        self.track_history: Dict[int, List[Dict[str, Any]]] = {}
        # Spatial persistence clusters: list of cluster dicts
        self.spatial_clusters: List[Dict[str, Any]] = []
        self.suppressed_events: List[Dict[str, Any]] = []
        self.telemetry_records: List[Dict[str, Any]] = []

        self.frame_width: Optional[float] = None
        self.frame_height: Optional[float] = None
        self.max_observed_w: float = 0.0
        self.max_observed_h: float = 0.0

    def log_suppressed(
        self,
        frame_idx: int,
        tracks: List[int],
        reason: str,
        dist_or_dca: float,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        val = round(float(dist_or_dca), 2)
        entry = {
            "frame": frame_idx,
            "tracks": list(tracks),
            "track_ids": list(tracks),
            "module": "single_vehicle",
            "reason": reason,
            "distance_or_dca": val,
            "metric_val": val,
        }
        if telemetry:
            entry["plausibility_telemetry"] = telemetry
        self.suppressed_events.append(entry)

    def reset(self) -> None:
        """Reset internal state between videos."""
        self.track_history.clear()
        self.spatial_clusters.clear()
        self.suppressed_events.clear()
        self.telemetry_records.clear()
        self.max_observed_w = 0.0
        self.max_observed_h = 0.0

    def compute_pre_impulse_velocity(
        self,
        hist: List[Dict[str, Any]],
        track: Track,
        window_s: float = 1.0,
    ) -> Tuple[float, int]:
        """Compute pre-impulse entry velocity v_entry and identify decel onset frame.

        1. Identify Deceleration Onset:
           Determine timestamp or frame index where continuous decelerative impulse began (a < -3.0 m/s^2).
        2. Compute Pre-Impulse Entry Velocity (v_entry):
           Calculate maximum velocity over 1.0s window strictly preceding deceleration onset:
           v_entry = max_{t in [t_onset - 1.0s, t_onset]} v(t)
        """
        if not hist:
            return 0.0, 0

        # Walk backwards from current frame to find start of continuous deceleration (a < -3.0 m/s^2)
        onset_idx = len(hist) - 1
        for i in range(len(hist) - 1, -1, -1):
            a_val = float(hist[i].get("accel_ms2", 0.0))
            if a_val < -3.0:
                onset_idx = i
            else:
                break

        onset_entry = hist[onset_idx]
        decel_onset_frame = onset_entry["frame_idx"]
        t_onset_s = onset_entry.get("timestamp_s", onset_entry["frame_idx"] / 30.0)

        # 1.0s window strictly preceding deceleration onset: t in [t_onset - 1.0s, t_onset]
        pre_speeds = []
        for e in hist[:onset_idx]:
            t_e = e.get("timestamp_s", e["frame_idx"] / 30.0)
            dt = t_onset_s - t_e
            if 0.0 <= dt <= window_s:
                pre_speeds.append(float(e.get("speed_kmh", 0.0)))

        if len(pre_speeds) >= 3:
            # Multi-frame consistency filter: require at least 3 consistent frames (filters 1-2 frame flicker spikes)
            rolling_mins = [min(pre_speeds[i:i + 3]) for i in range(len(pre_speeds) - 2)]
            v_entry = float(max(rolling_mins))
        elif pre_speeds:
            v_entry = float(np.median(pre_speeds))
        else:
            track_spd = getattr(track, "max_recent_speed_kmh", 0.0)
            v_entry = float(max(track_spd, onset_entry.get("speed_kmh", 0.0)))

        return v_entry, decel_onset_frame

    def evaluate_kinematic_plausibility(
        self,
        track: Track,
        hist: List[Dict[str, Any]],
        clamped_a: float,
        curr_spd: float,
        frame_idx: int,
        pts_ns: int,
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """Plausibility Decision Tree using pre-impulse entry velocity (v_entry)."""
        ts_s = pts_ns * 1e-9 if pts_ns > 0 else frame_idx / 30.0
        abs_accel = abs(clamped_a)

        if abs_accel >= 12.0 and clamped_a < 0.0:
            v_entry_kmh, decel_onset_frame = self.compute_pre_impulse_velocity(hist, track, window_s=1.0)

            if v_entry_kmh >= 15.0:
                # Significant kinetic energy prior to deceleration onset:
                # Rapid velocity collapse is physically consistent with a high-energy crash.
                is_plausible = True
                rejection_reason = None
                verdict = "ACCEPTED_PHYSICAL_IMPACT"
                tau_rest = (v_entry_kmh / 3.6) / (abs_accel + 1e-6)
            else:
                # Low pre-impulse speed: reject stationary/Kalman jitter (< 15.0 km/h cannot have high-speed crash)
                v_entry_mps = v_entry_kmh / 3.6
                tau_rest = v_entry_mps / (abs_accel + 1e-6)
                is_plausible = False
                verdict = "REJECTED_JITTER"
                rejection_reason = f"low_ventry_jitter (v_entry={v_entry_kmh:.1f}km/h < 15.0km/h)"
        else:
            v_entry_kmh = max(curr_spd, getattr(track, "max_recent_speed_kmh", 0.0))
            decel_onset_frame = frame_idx
            tau_rest = (v_entry_kmh / 3.6) / (abs_accel + 1e-6)
            is_plausible = True
            rejection_reason = None
            verdict = "ACCEPTED_PHYSICAL_IMPACT"

        telemetry = {
            "track_id": getattr(track, "track_id", getattr(track, "id", None)),
            "frame_idx": frame_idx,
            "timestamp_s": round(ts_s, 3),
            "decel_onset_frame": decel_onset_frame,
            "v_entry_kmh": round(v_entry_kmh, 2),
            "v_curr_kmh": round(curr_spd, 2),
            "accel_mps2": round(clamped_a, 2),
            "tau_rest_s": round(tau_rest, 3),
            "plausibility_verdict": verdict,
            "rejection_reason": rejection_reason,
        }
        return is_plausible, rejection_reason, telemetry

    def _estimate_metric_acceleration(self, bbox_xywh: Tuple[float, float, float, float], raw_accel: float) -> float:
        """Estimate physically scaled acceleration in m/s^2 from perspective or homography."""
        if self.is_camera_calibrated:
            return raw_accel
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_accel * scale

    def _estimate_metric_speed(self, bbox_xywh: Tuple[float, float, float, float], raw_speed: float) -> float:
        """Estimate physically scaled speed in km/h."""
        if self.is_camera_calibrated:
            return raw_speed
        w, h = bbox_xywh[2], bbox_xywh[3]
        char_px = max(15.0, max(w, h))
        scale = 4.5 / char_px
        return raw_speed * scale

    def set_resolution(self, width: float, height: float) -> None:
        """Set image resolution for strict 4-boundary sensor edge clipping and default roadway polygon."""
        self.frame_width = float(width)
        self.frame_height = float(height)
        if self.roadway_polygon is None:
            self.roadway_polygon = [
                (0.0, float(height)),
                (float(width), float(height)),
                (float(width) * 0.75, float(height) * 0.35),
                (float(width) * 0.25, float(height) * 0.35),
            ]

    def set_roadway_polygon(self, roadway_polygon: List[Tuple[float, float]]) -> None:
        """Set roadway polygon explicitly."""
        self.roadway_polygon = roadway_polygon

    def _is_edge_clipped(self, bbox_xywh: Tuple[float, float, float, float]) -> bool:
        """Reject detections clipped by camera sensor boundaries."""
        bx, by, bw, bh = bbox_xywh
        margin = self.edge_margin_px
        if margin <= 0.0:
            return False
        if bx < margin:
            return True
        if self.frame_width and (bx + bw) >= (self.frame_width - margin):
            return True
        if by < margin:
            return True
        if self.frame_height and (by + bh) >= (self.frame_height - margin):
            return True
        return False

    def evaluate(
        self,
        frame_idx: int,
        pts_ns: int,
        video_id: str,
        tracks: Dict[int, Track],
        kinematics: Dict[int, TrackKinematics],
        convergences: List[TrajectoryConvergence],
        lane_manager: LaneManager,
    ) -> List[EventCandidate]:
        candidates: List[EventCandidate] = []
        active_tids: Set[int] = set()

        if frame_idx < self.video_warmup_frames:
            return candidates

        for tid, tr in tracks.items():
            eff_label = tr.get_effective_label().lower() if hasattr(tr, "get_effective_label") else tr.class_label.lower()
            if eff_label not in self.VEHICLE_LABELS or eff_label == "candidate_vehicle_recovered":
                continue
            if getattr(tr, "audit_reclassified", False) or getattr(tr, "fused_class", "") == "candidate_vehicle_recovered" or getattr(tr, "is_tentative_rescued", False):
                continue

            k = kinematics.get(tid)
            if not k:
                continue

            active_tids.add(tid)
            bbox = tr.current_bbox_xywh
            w, h = bbox[2], bbox[3]
            char_dim = max(15.0, max(w, h))
            ar = float(w / max(1.0, h))

            # 1. Reject tiny distant background noise where bounding box jitter induces fake kinematics
            if min(w, h) < 12.0 or char_dim < 25.0:
                continue

            # 2. Reject edge-clipped boundary tracks (all 4 sensor boundaries)
            if self._is_edge_clipped(bbox):
                continue

            # Ground coordinates and calibration check
            gx, gy = k.ground_pos
            is_ground_valid = getattr(k, "is_ground_valid", True) and (abs(gx) <= 150.0 and -50.0 <= gy <= 250.0)

            # Pixel kinematics
            px_spd = getattr(k, "pixel_speed", 0.0)
            px_acc = getattr(k, "pixel_acceleration", 0.0)
            if px_spd == 0.0:
                from road_surveillance.geometry.spatial_metrics import compute_pixel_kinematics
                px_spd, px_acc = compute_pixel_kinematics(tr, pts_ns)

            # Metric kinematics only when calibrated & ground point is geometrically valid
            if self.is_camera_calibrated and is_ground_valid:
                metric_speed = k.speed_kmh
                metric_a = k.acceleration_mag
            else:
                # Body-scaled pixel kinematics
                metric_speed = (px_spd / char_dim) * 4.5 * 3.6
                metric_a = (px_acc / char_dim) * 4.5

            entry = {
                "frame_idx": frame_idx,
                "timestamp_s": pts_ns * 1e-9 if pts_ns > 0 else frame_idx / 30.0,
                "bbox": bbox,
                "ar": ar,
                "heading": k.heading_deg,
                "speed_kmh": metric_speed,
                "accel_ms2": metric_a,
                "ground_pos": k.ground_pos,
                "track_age": tr.track_age,
            }

            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append(entry)
            hist = self.track_history[tid]

            # 1. Track Maturity Gate
            if tr.track_age < self.min_track_age or len(hist) < self.min_track_age or len(tr.ground_history) < self.min_track_age:
                continue

            # 2. Re-acquisition Settlement Gate (tracks recovering from coasting/lost must have >= 3 matched frames)
            if not getattr(tr, "is_reacquisition_settled", True):
                continue

            if len(hist) > 60:
                hist.pop(0)

            prev_entry = hist[-2]

            # 3. Physically plausible bounds: positive forward acceleration is traction-bounded (<= 5.0 m/s^2),
            # while severe collision impact deceleration can reach -45.0 m/s^2.
            # Abnormal positive acceleration spikes across 1-2 frames are treated as detector flickers.
            MAX_FORWARD_ACCEL = 5.0
            clamped_a = float(np.clip(metric_a, -self.max_physical_decel, MAX_FORWARD_ACCEL))

            # Signal 1: Severe barrier impact deceleration
            hist_speeds = [h_entry["speed_kmh"] for h_entry in hist[:-1]]
            if len(hist_speeds) >= 3:
                # Multi-frame consistency filter: require >= 3-4 consecutive frames of consistent motion
                rolling_mins = [min(hist_speeds[i:i + 3]) for i in range(len(hist_speeds) - 2)]
                max_recent_spd = float(max(rolling_mins))
            elif hist_speeds:
                max_recent_spd = float(np.median(hist_speeds))
            else:
                max_recent_spd = 0.0
            signal_severe_impact = bool(-self.max_physical_decel <= clamped_a <= self.severe_impact_decel and max_recent_spd >= 12.0)

            # Signal 2: Moderate Deceleration
            signal_moderate_decel = bool(self.severe_impact_decel < clamped_a < self.decel_threshold)

            # Signal 3: Sudden Stop / Kinetic Energy Loss
            curr_spd = metric_speed
            spd_drop = max_recent_spd - curr_spd
            signal_sudden_stop = bool(max_recent_spd >= 12.0 and curr_spd <= 6.0 and spd_drop >= 10.0)

            # Signal 4: Abnormal Yaw / Trajectory Angular Deviation
            dh = abs(entry["heading"] - prev_entry["heading"])
            if dh > 180.0:
                dh = 360.0 - dh
            signal_abnormal_yaw = bool(dh >= self.yaw_rate_thresh and max_recent_spd >= 8.0)

            # Signal 5: Rollover / Aspect Ratio Inversion
            prev_ar = prev_entry["ar"]
            ar_ratio = abs(ar - prev_ar) / max(0.1, prev_ar)
            signal_rollover = bool(ar_ratio >= self.ar_change_thresh)

            # Signal 6: Continuous Composite Evidence Scoring (Iteration M)
            has_lead = check_has_lead_vehicle(tid, tracks, kinematics)
            road_poly = getattr(lane_manager, "roadway_polygon", self.roadway_polygon)
            s_single, detail = compute_single_vehicle_evidence(
                track=tr,
                roadway_poly=road_poly,
                homography=getattr(lane_manager, "homography", None),
                kinematics=k,
                has_lead_vehicle=has_lead,
                heading_deviation_deg=dh,
                aspect_ratio_delta=ar_ratio,
            )
            signal_composite = bool(s_single >= self.tau_single)

            # Boundary proximity check
            d_edge_m = float(detail.get("d_edge_m", 99.0))
            s_bound = float(detail.get("s_bound", 0.0))
            is_near_boundary = (d_edge_m <= 2.5 or s_bound > 0.0)

            # 4. SINGLE-VEHICLE ANOMALY GATE
            # Allow barrier_strike when:
            # - severe deceleration / impulse: delta_a <= -14.0 m/s^2 (or |a| >= 14.0 m/s^2)
            # - kinetic arrest: speed drops to <= 6.0 km/h with delta_speed >= 10.0 km/h (or signal_sudden_stop)
            # - near roadway boundary: close to barrier/edge
            # - OR evidence-gated shoulder strike (Task R.2)
            is_shoulder_strike = bool(
                detail.get("shoulder_admitted", False)
                and (signal_severe_impact or signal_moderate_decel or signal_sudden_stop or clamped_a <= -4.0)
                and clamped_a < 0.0
                and (curr_spd <= 15.0 or signal_sudden_stop or spd_drop >= 8.0)
            )
            # Pre-existing stationary vehicle guard:
            # If track was already stationary (speed <= 5.0 km/h) over recent history,
            # it is a parked vehicle or static obstacle; it cannot crash into a barrier from rest.
            recent_speeds = [h_entry["speed_kmh"] for h_entry in hist]
            is_pre_stationary = bool(len(recent_speeds) >= 8 and float(np.median(recent_speeds[:-1])) <= 5.0)

            is_barrier_strike = bool(
                not is_pre_stationary
                and (
                    (
                        (clamped_a <= self.severe_impact_decel)
                        and (curr_spd <= 6.0 or signal_sudden_stop)
                        and (spd_drop >= 10.0 and max_recent_spd >= 15.0)
                        and is_near_boundary
                        and (s_single >= 0.35 or (s_single >= 0.25 and signal_sudden_stop))
                    )
                    or is_shoulder_strike
                )
            )

            is_anomaly = False
            immediate_trigger = False
            active_signals: List[str] = []
            plaus_tel: Optional[Dict[str, Any]] = None

            if is_barrier_strike:
                # Upstream Kinematic Plausibility Gate (Task Directive 1 & 2)
                is_plausible, rej_reason, plaus_tel = self.evaluate_kinematic_plausibility(
                    track=tr,
                    hist=hist,
                    clamped_a=clamped_a,
                    curr_spd=curr_spd,
                    frame_idx=frame_idx,
                    pts_ns=pts_ns,
                )
                self.telemetry_records.append(plaus_tel)
                if not is_plausible:
                    self.log_suppressed(frame_idx, [tid], rej_reason or "low_ventry_jitter", clamped_a, telemetry=plaus_tel)
                    continue
                is_anomaly = True
                immediate_trigger = True
                active_signals.append("severe_barrier_impact")
            elif s_single >= 0.55 or (
                s_single >= 0.30
                and signal_sudden_stop
                and not has_lead
                and (signal_severe_impact or clamped_a <= -10.0)
                and max_recent_spd >= 15.0
                and not is_pre_stationary
            ):
                # For non-barrier single-vehicle incidents, require post-event kinetic dissipation
                is_dissipating = (curr_spd <= 20.0 or signal_sudden_stop or getattr(k, "stationary_duration_sec", 0.0) >= 0.25)
                if not is_dissipating:
                    self.log_suppressed(frame_idx, [tid], "single_vehicle_unsubstantiated", s_single)
                    continue

                # Upstream Kinematic Plausibility Gate (Task Directive 1 & 2)
                is_plausible, rej_reason, plaus_tel = self.evaluate_kinematic_plausibility(
                    track=tr,
                    hist=hist,
                    clamped_a=clamped_a,
                    curr_spd=curr_spd,
                    frame_idx=frame_idx,
                    pts_ns=pts_ns,
                )
                self.telemetry_records.append(plaus_tel)
                if not is_plausible:
                    self.log_suppressed(frame_idx, [tid], rej_reason or "low_ventry_jitter", clamped_a, telemetry=plaus_tel)
                    continue

                # Require genuine anomaly evidence (rollover / spin-out / decel stop)
                if signal_rollover and max_recent_spd >= 10.0 and (curr_spd <= 18.0 or signal_sudden_stop) and (clamped_a <= -4.0 or signal_sudden_stop):
                    is_anomaly = True
                    active_signals.append("rollover")
                elif signal_abnormal_yaw and (signal_severe_impact or signal_moderate_decel or (spd_drop >= 8.0 and clamped_a <= -3.0)) and (curr_spd <= 20.0 or signal_sudden_stop):
                    is_anomaly = True
                    active_signals.append("yaw")
                elif (signal_moderate_decel and curr_spd <= 15.0) or signal_sudden_stop:
                    is_anomaly = True
                    active_signals.append("decel_stop")
                elif s_single >= 0.70 and (curr_spd <= 15.0 or signal_sudden_stop) and (clamped_a <= -2.0 or signal_sudden_stop):
                    is_anomaly = True
                    active_signals.append(f"composite({s_single:.2f})")
                else:
                    self.log_suppressed(frame_idx, [tid], "single_vehicle_unsubstantiated", s_single)
                    continue

                # Scope-Isolated Low-Speed Dynamic Gate
                # Apply exclusively to dynamic single-vehicle anomalies (yaw, rollover, spin-out, composite)
                if any(any(d in sig for d in DYNAMIC_ANOMALY_TYPES) for sig in active_signals):
                    track_max_spd = getattr(tr, "max_recent_speed_kmh", 0.0)
                    if track_max_spd <= 0.0:
                        track_max_spd = max_recent_spd
                    if track_max_spd < 15.0:
                        self.log_suppressed(frame_idx, [tid], "low_speed_dynamic_floor_suppressed", track_max_spd)
                        continue

                if s_single >= 0.70:
                    immediate_trigger = True
            else:
                self.log_suppressed(frame_idx, [tid], "single_vehicle_unsubstantiated", s_single)

            if not is_anomaly:
                continue

            centroid = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)

            # Temporal persistence verification
            matched = False
            triggered = immediate_trigger

            if not triggered:
                for cluster in self.spatial_clusters:
                    if cluster["track_id"] == tid and (frame_idx - cluster["last_frame"]) <= 3:
                        cluster["persistence"] += 1
                        cluster["last_frame"] = frame_idx
                        cluster["box"] = bbox
                        matched = True
                        if cluster["persistence"] >= self.min_persistence_frames:
                            triggered = True
                            break

                if not matched:
                    self.spatial_clusters.append({
                        "track_id": tid,
                        "centroid": centroid,
                        "box": bbox,
                        "last_frame": frame_idx,
                        "persistence": 1,
                    })
                    if self.min_persistence_frames <= 1:
                        triggered = True

            if triggered:
                conf = 0.95 if is_barrier_strike else (0.90 if immediate_trigger else 0.75)
                cand_type = "barrier_strike" if is_barrier_strike else "single_vehicle_anomaly"
                cand_signals = {
                    "anomaly_type": "+".join(active_signals),
                    "composite_score": float(s_single),
                    "s_theta": float(detail.get("s_theta", 0.0)),
                    "s_decel": float(detail.get("s_decel", 0.0)),
                    "s_bound": float(detail.get("s_bound", 0.0)),
                    "s_ar": float(detail.get("s_ar", 0.0)),
                    "theta_deg": float(detail.get("theta_deg", 0.0)),
                    "decel_mag": float(detail.get("decel_mag", 0.0)),
                    "d_edge_m": float(detail.get("d_edge_m", 0.0)),
                    "delta_ar": float(detail.get("delta_ar", 0.0)),
                    "shoulder_admitted": bool(detail.get("shoulder_admitted", False)),
                    "v_dot_n": float(detail.get("v_dot_n", 0.0)),
                    "metric_accel_ms2": float(clamped_a),
                    "metric_speed_kmh": float(metric_speed),
                    "max_recent_speed_kmh": float(max_recent_spd),
                    "heading_deg": float(k.heading_deg),
                    "yaw_delta_deg": float(dh),
                    "aspect_ratio": float(ar),
                    "aspect_ratio_ratio": float(ar_ratio),
                }
                if plaus_tel:
                    cand_signals["kinematic_plausibility"] = plaus_tel
                    cand_signals["v_entry_kmh"] = plaus_tel["v_entry_kmh"]
                    cand_signals["decel_onset_frame"] = plaus_tel["decel_onset_frame"]
                    cand_signals["tau_rest_s"] = plaus_tel["tau_rest_s"]
                    cand_signals["plausibility_verdict"] = plaus_tel["plausibility_verdict"]
                else:
                    cand_signals["v_entry_kmh"] = float(max_recent_spd)

                candidates.append(
                    EventCandidate(
                        event_type=cand_type,
                        video_id=video_id,
                        frame_idx=frame_idx,
                        timestamp_ns=pts_ns,
                        track_ids=[tid],
                        primary_track_id=tid,
                        confidence=conf,
                        signals=cand_signals,
                        ground_coordinates=k.ground_pos,
                        lane_id=k.lane_id,
                        description=(
                            f"Single-vehicle anomaly on track {tid} ({tr.class_label}): "
                            f"type={'+'.join(active_signals)}, S_single={s_single:.2f}, "
                            f"accel={metric_a:.1f}m/s^2, spd={metric_speed:.1f}km/h"
                        ),
                    )
                )

        # Cleanup disappeared tracks
        for t in list(self.track_history.keys()):
            if t not in active_tids:
                del self.track_history[t]

        # Prune expired clusters
        self.spatial_clusters = [
            c for c in self.spatial_clusters if (frame_idx - c["last_frame"]) <= 10
        ]

        return candidates
