"""Episode Consolidation and VLM Dispatch Control Module.

Iteration P: Daytime Specificity Hardening (P.3)
- Restores a minimum 10 s cooldown for identical candidate track IDs.
- Consolidates duplicate candidate proposals into coherent physical incident episodes.
- Selects the peak saliency candidate per episode for VLM dispatch.
- Measures the full progression pipeline:
  Raw Proposals -> Consolidated Episodes -> VLM Calls -> Confirmed Incidents
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConsolidatedEpisode:
    """Represents a consolidated spatiotemporal incident episode."""
    episode_id: str
    video_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    primary_event_type: str
    involved_track_ids: List[int]
    raw_candidates: List[Dict[str, Any]]
    peak_candidate: Dict[str, Any]
    peak_frame: int
    peak_score: float
    is_in_gt_window: bool = False
    vlm_dispatched: bool = False
    vlm_result: Optional[Dict[str, Any]] = None
    is_confirmed: bool = False
    dispatch_suppressed: bool = False
    suppression_reason: Optional[str] = None
    cooldown_telemetry: Optional[Dict[str, Any]] = None
    initial_trigger_timestamp_s: float = 0.0
    settling_telemetry: List[Dict[str, Any]] = field(default_factory=list)
    state: str = "ACTIVE"
    primary_actor_pair: Optional[Tuple[int, int]] = None
    secondary_actor_pairs: List[Tuple[int, int]] = field(default_factory=list)
    primary_impact_centroid: Optional[Tuple[float, float]] = None
    primary_impact_pixel_centroid: Optional[Tuple[float, float]] = None
    primary_impact_char_dim: float = 30.0
    aftermath_result: Optional[Any] = None

    @property
    def track_ids(self) -> List[int]:
        return self.involved_track_ids

    @property
    def participating_track_ids(self) -> List[int]:
        return self.involved_track_ids

    @property
    def spatial_centroid(self) -> Any:
        from road_surveillance.events.base import SpatialPoint
        loc = self.peak_candidate.get("ground_coordinates", (0.0, 0.0))
        return SpatialPoint(loc[0], loc[1])

    def append_settling_telemetry(self, cand: Dict[str, Any]) -> None:
        """Absorbs post-impact coordinate jitter/settling into primary crash episode."""
        self.state = "POST_IMPACT_SETTLING"
        c_frame = cand["frame"]
        c_time = cand.get("timestamp_s", c_frame / 30.0)
        self.end_frame = max(self.end_frame, c_frame)
        self.end_time_s = max(self.end_time_s, c_time)
        for tid in cand.get("track_ids", []):
            if tid not in self.involved_track_ids:
                self.involved_track_ids.append(tid)
        if cand not in self.raw_candidates:
            self.raw_candidates.append(cand)
        telemetry = {
            "state": "POST_IMPACT_SETTLING",
            "frame": c_frame,
            "timestamp_s": c_time,
            "track_ids": cand.get("track_ids", []),
            "speed_kmh": float(cand.get("signals", {}).get("metric_speed_kmh", cand.get("signals", {}).get("curr_spd", cand.get("speed_kmh", 0.0)))),
            "ground_coordinates": cand.get("ground_coordinates", (0.0, 0.0)),
            "signals": cand.get("signals", {}),
        }
        self.settling_telemetry.append(telemetry)


class ActorDispatchTracker:
    """Task S.A.3: Actor-Level Dispatch Suppression.

    Maintains a 10.0-second experimental dispatch cooldown keyed on participating actor track IDs.
    Distinguishes duplicate dispatch suppression from physical incident identity:
    Does NOT merge two genuine incidents merely because the same track IDs occur within cooldown.
    Logs every suppression as 'actor_cooldown_active' with actor IDs, timestamp, previous dispatch time, and delta time.
    """
    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_dispatch_time_by_actor: Dict[int, float] = {}
        self.suppression_logs: List[Dict[str, Any]] = []

    def should_suppress_dispatch(
        self,
        actor_ids: List[int],
        timestamp_s: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if any participating actor is under active dispatch cooldown."""
        for tid in actor_ids:
            if tid in self._last_dispatch_time_by_actor:
                prev_time = self._last_dispatch_time_by_actor[tid]
                dt = timestamp_s - prev_time
                if 0.0 <= dt < self.cooldown_seconds:
                    log_entry = {
                        "event": "actor_cooldown_active",
                        "actor_ids": sorted(list(actor_ids)),
                        "suppressed_actor_id": tid,
                        "timestamp_s": round(timestamp_s, 3),
                        "previous_dispatch_time_s": round(prev_time, 3),
                        "delta_time_s": round(dt, 3),
                        "cooldown_limit_s": self.cooldown_seconds,
                    }
                    self.suppression_logs.append(log_entry)
                    return True, log_entry

        return False, None

    def record_dispatch(self, actor_ids: List[int], timestamp_s: float) -> None:
        """Records dispatch timestamp for participating actors."""
        for tid in actor_ids:
            self._last_dispatch_time_by_actor[tid] = timestamp_s

    def reset(self) -> None:
        self._last_dispatch_time_by_actor.clear()
        self.suppression_logs.clear()

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        return self.raw_candidates

    @property
    def peak_confidence(self) -> float:
        return float(self.peak_candidate.get("confidence", 0.0))

    @property
    def attached_near_misses(self) -> List[Dict[str, Any]]:
        misses = []
        for c in self.raw_candidates:
            misses.extend(c.get("signals", {}).get("near_misses", []))
        return misses


def calculate_candidate_saliency(
    candidate: Dict[str, Any],
    frame_pts_map: Optional[Dict[int, float]] = None,
) -> float:
    """Computes a physical saliency score to identify the peak impact frame.

    Higher scores represent more acute physical conflict:
    - High longitudinal deceleration / impulse
    - Tight bounding box IoU / physical contact
    - Close inter-vehicle distance
    - High composite single-vehicle score
    - Subtractive PTS-native temporal decay penalty based on continuous elapsed time from kinetic onset.
    """
    signals = candidate.get("signals", {})
    score = 0.0

    # 1. Acceleration / impulse magnitude (up to 50 pts)
    accel = abs(float(signals.get("accel_ms2", signals.get("accel_1_ms2", signals.get("metric_accel_ms2", 0.0)))))
    score += min(50.0, accel * 3.0)

    # 2. Bounding box IoU (physical contact, up to 40 pts)
    iou = float(signals.get("bbox_iou", 0.0))
    score += iou * 40.0

    # 3. Inter-vehicle distance inversion (closer = higher score, up to 30 pts)
    dist = float(signals.get("ground_distance_m", signals.get("distance_m", signals.get("dca_m", 10.0))))
    if dist < 10.0:
        score += max(0.0, (5.0 - dist) * 6.0)

    # 4. Composite score for single-vehicle anomaly (up to 30 pts)
    comp = float(signals.get("composite_score", signals.get("s_single", 0.0)))
    score += comp * 30.0

    # 5. Type priority: confirmed multi-vehicle collision or barrier strike gets high priority
    c_type = candidate.get("event_type", "")
    if c_type in ("collision_candidate", "barrier_strike"):
        score += 25.0
    elif c_type in ("wrong_way_hazard", "wrong_way_vehicle"):
        score += 20.0

    # 6. Candidate confidence
    conf = float(candidate.get("confidence", 0.5))
    score += conf * 10.0

    # 7. Subtractive PTS-Native Temporal Saliency Decay Penalty
    c_frame = candidate.get("frame", 0)
    if "timestamp_s" in candidate:
        t_cand_pts = float(candidate["timestamp_s"])
    elif frame_pts_map is not None and c_frame in frame_pts_map:
        t_cand_pts = float(frame_pts_map[c_frame])
    elif "timestamp_ns" in candidate:
        t_cand_pts = float(candidate["timestamp_ns"]) * 1e-9
    else:
        fps = float(candidate.get("fps", signals.get("fps", 30.0)))
        t_cand_pts = c_frame / fps if fps > 0 else 0.0

    # Kinetic onset extraction
    t_onset_pts = None
    if "decel_onset_pts" in signals:
        t_onset_pts = float(signals["decel_onset_pts"])
    elif "decel_onset_pts" in candidate:
        t_onset_pts = float(candidate["decel_onset_pts"])
    elif "kinematic_plausibility" in signals and "decel_onset_pts" in signals["kinematic_plausibility"]:
        t_onset_pts = float(signals["kinematic_plausibility"]["decel_onset_pts"])
    elif "decel_onset_frame" in signals:
        onset_frame = int(signals["decel_onset_frame"])
        if frame_pts_map is not None and onset_frame in frame_pts_map:
            t_onset_pts = float(frame_pts_map[onset_frame])
        elif candidate.get("frame") and candidate.get("timestamp_s") and candidate["frame"] > 0:
            cand_fps = float(candidate["frame"]) / float(candidate["timestamp_s"])
            t_onset_pts = onset_frame / cand_fps
        else:
            fps = float(candidate.get("fps", signals.get("fps", 30.0)))
            t_onset_pts = onset_frame / fps if fps > 0 else 0.0
    elif "decel_onset_frame" in candidate:
        onset_frame = int(candidate["decel_onset_frame"])
        if frame_pts_map is not None and onset_frame in frame_pts_map:
            t_onset_pts = float(frame_pts_map[onset_frame])
        elif candidate.get("frame") and candidate.get("timestamp_s") and candidate["frame"] > 0:
            cand_fps = float(candidate["frame"]) / float(candidate["timestamp_s"])
            t_onset_pts = onset_frame / cand_fps
        else:
            fps = float(candidate.get("fps", signals.get("fps", 30.0)))
            t_onset_pts = onset_frame / fps if fps > 0 else 0.0
    elif "kinematic_plausibility" in signals and "decel_onset_frame" in signals["kinematic_plausibility"]:
        onset_frame = int(signals["kinematic_plausibility"]["decel_onset_frame"])
        if frame_pts_map is not None and onset_frame in frame_pts_map:
            t_onset_pts = float(frame_pts_map[onset_frame])
        elif candidate.get("frame") and candidate.get("timestamp_s") and candidate["frame"] > 0:
            cand_fps = float(candidate["frame"]) / float(candidate["timestamp_s"])
            t_onset_pts = onset_frame / cand_fps
        else:
            fps = float(candidate.get("fps", signals.get("fps", 30.0)))
            t_onset_pts = onset_frame / fps if fps > 0 else 0.0

    if t_onset_pts is not None:
        delta_t_delay_s = abs(t_cand_pts - t_onset_pts)
    else:
        delta_t_delay_s = 0.0
        logger.debug(f"Candidate @ F{c_frame}: kinematic onset unavailable, falling back to delta_t_delay_s = 0.0s")

    k_decay = 45.0  # pts/second (equivalent to 1.50 pts/frame at nominal 30 fps)
    temporal_penalty = k_decay * delta_t_delay_s
    score -= temporal_penalty

    return score


def _standard_relational_merge(
    ep: ConsolidatedEpisode,
    cand: Dict[str, Any],
    max_dt: float = 1.8,
    fps: Optional[float] = None,
) -> bool:
    """Standard global relational consolidation with Actor Consistency (Proposal B)."""
    c_frame = cand["frame"]
    c_time = cand.get("timestamp_s", c_frame / 30.0)
    c_tracks = set(cand.get("track_ids", []))
    c_type = cand.get("event_type", "")

    # Invariant: Do NOT merge wrong_way_hazard with barrier_strike / collision!
    is_c_ww = c_type in ("wrong_way_hazard", "wrong_way_vehicle")
    is_ep_ww = ep.primary_event_type in ("wrong_way_hazard", "wrong_way_vehicle")
    is_c_impact = c_type in ("barrier_strike", "single_vehicle_anomaly", "collision_candidate")
    is_ep_impact = ep.primary_event_type in ("barrier_strike", "single_vehicle_anomaly", "collision_candidate")

    if (is_c_ww and is_ep_impact) or (is_c_impact and is_ep_ww):
        return False

    # Dynamic FPS calculation for max_frame_gap (Safeguard 3)
    if fps is None:
        cand_fps = cand.get("fps", cand.get("signals", {}).get("fps"))
        if cand_fps is not None and cand_fps > 0:
            fps = float(cand_fps)
        elif c_time > 0 and c_frame > 0:
            fps = float(c_frame / c_time)
        else:
            fps = 30.0
    max_frame_gap = int(round(max_dt * fps))

    # Authoritative PTS time delta check (Safeguard 3) & frame gap disjunction
    time_gap = abs(c_time - ep.end_time_s)
    frame_gap = abs(c_frame - ep.end_frame)
    if time_gap > max_dt or frame_gap > max_frame_gap:
        return False

    # Proposal B: Actor Consistency Check for collision episodes
    if ep.primary_actor_pair is not None:
        primary_set = set(ep.primary_actor_pair)

        # Disjoint Actor Rejection: candidate shares 0 actors with primary pair
        if not c_tracks.intersection(primary_set):
            return False

        # If candidate is a collision candidate or has >= 2 tracks:
        if c_type == "collision_candidate" or len(c_tracks) >= 2:
            # Case A: Exact Pair Continuity
            if c_tracks == primary_set:
                pass  # Permitted exact continuity
            # Case B: Secondary Impact / Chain Reaction (|P_cand ∩ P_primary| >= 1)
            else:
                # Spatial proximity check to primary impact site
                c_center = cand.get("ground_coordinates", (0.0, 0.0))
                if c_center == (0.0, 0.0):
                    c_center = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
                ep_center = ep.primary_impact_centroid or ep.peak_candidate.get("ground_coordinates", (0.0, 0.0))
                if ep_center == (0.0, 0.0):
                    ep_center = ep.peak_candidate.get("signals", {}).get("ground_coordinates", (0.0, 0.0))

                has_spatial_proximity = False
                # If calibrated ground coordinates are available:
                if c_center != (0.0, 0.0) and ep_center != (0.0, 0.0):
                    spatial_dist = float(np.hypot(c_center[0] - ep_center[0], c_center[1] - ep_center[1]))
                    if spatial_dist <= 4.0:
                        has_spatial_proximity = True
                else:
                    # Safeguard 4: Normalized 2D fallback for uncalibrated camera
                    c_px = cand.get("signals", {}).get("pixel_centroid")
                    ep_px = ep.primary_impact_pixel_centroid or ep.peak_candidate.get("signals", {}).get("pixel_centroid")
                    char_dim = ep.primary_impact_char_dim or float(cand.get("signals", {}).get("char_dim", 30.0))
                    iou = float(cand.get("signals", {}).get("bbox_iou", 0.0))

                    if c_px is not None and ep_px is not None and char_dim > 0:
                        px_dist = float(np.hypot(c_px[0] - ep_px[0], c_px[1] - ep_px[1]))
                        norm_pileup_dist = px_dist / char_dim
                        if norm_pileup_dist <= 1.5 or iou > 0.0:
                            has_spatial_proximity = True
                    elif iou > 0.0:
                        has_spatial_proximity = True

                if not has_spatial_proximity:
                    return False
    else:
        # Non-collision episodes: require track overlap with involved tracks
        ep_tracks = set(ep.involved_track_ids)
        track_overlap = bool(c_tracks.intersection(ep_tracks))
        if not track_overlap:
            return False

        # Ground distance between trigger spatial centers <= 6.0 meters
        c_center = cand.get("ground_coordinates", (0.0, 0.0))
        if c_center == (0.0, 0.0):
            c_center = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
        ep_center = ep.peak_candidate.get("ground_coordinates", (0.0, 0.0))
        if ep_center == (0.0, 0.0):
            ep_center = ep.peak_candidate.get("signals", {}).get("ground_coordinates", (0.0, 0.0))

        if c_center != (0.0, 0.0) and ep_center != (0.0, 0.0):
            spatial_dist = float(np.hypot(c_center[0] - ep_center[0], c_center[1] - ep_center[1]))
            if spatial_dist > 6.0:
                return False

    # 4. Compatible event lineage
    is_compat = (
        (c_type == ep.primary_event_type)
        or (c_type == "collision_candidate" and ep.primary_event_type in ("single_vehicle_anomaly", "barrier_strike"))
        or (ep.primary_event_type == "collision_candidate" and c_type in ("single_vehicle_anomaly", "barrier_strike"))
        or (c_type in ("barrier_strike", "single_vehicle_anomaly") and ep.primary_event_type in ("barrier_strike", "single_vehicle_anomaly"))
    )
    return is_compat


def should_consolidate_into_active_episode(
    ep: ConsolidatedEpisode,
    cand: Dict[str, Any],
    fps: Optional[float] = None,
) -> bool:
    """Task 3: Stateful Post-Impact Settlement Extension."""
    # 1. Standard global relational consolidation (frozen at 1.8s)
    if _standard_relational_merge(ep, cand, max_dt=1.8, fps=fps):
        return True

    # 2. Stateful Post-Impact Settlement (Barrier Strike continuation)
    ep_anom = ep.peak_candidate.get("signals", {}).get("anomaly_type", "") if ep.peak_candidate else ""
    if ep.primary_event_type in {"barrier_strike", "severe_barrier_impact"} or "barrier" in ep_anom:
        c_time = cand.get("timestamp_s", cand["frame"] / 30.0)
        time_since_initial = c_time - ep.initial_trigger_timestamp_s

        is_same_actor = bool(set(ep.involved_track_ids) & set(cand.get("track_ids", [])))

        c_spd = float(cand.get("signals", {}).get("metric_speed_kmh", cand.get("signals", {}).get("curr_spd", cand.get("speed_kmh", 0.0))))
        is_stationary_or_crawling = c_spd < 5.0

        c_center = cand.get("ground_coordinates", (0.0, 0.0))
        if c_center == (0.0, 0.0):
            c_center = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
        ep_center = ep.peak_candidate.get("ground_coordinates", (0.0, 0.0))
        if ep_center == (0.0, 0.0):
            ep_center = ep.peak_candidate.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
        if c_center != (0.0, 0.0) and ep_center != (0.0, 0.0):
            spatial_dist = float(np.hypot(c_center[0] - ep_center[0], c_center[1] - ep_center[1]))
            in_same_barrier_zone = spatial_dist <= 8.0
        else:
            in_same_barrier_zone = True

        if (
            is_same_actor
            and in_same_barrier_zone
            and is_stationary_or_crawling
            and time_since_initial <= 5.0
        ):
            ep.append_settling_telemetry(cand)
            return True

    return False


def consolidate_candidates_into_episodes(
    raw_candidates: List[Dict[str, Any]],
    video_id: str,
    cooldown_seconds: float = 10.0,
    max_gap_frames: int = 60,
    gt_frame: Optional[int] = None,
    temporal_tolerance: int = 45,
    fps: Optional[float] = None,
    frame_pts_map: Optional[Dict[int, float]] = None,
) -> List[ConsolidatedEpisode]:
    """Clusters raw frame-by-frame candidate proposals into consolidated incident episodes.

    Enforces:
    - Minimum 10 s cooldown for identical track IDs
    - Temporal and spatial clustering with Actor Consistency (Proposal B)
    - Peak candidate selection
    - In-window vs out-of-window classification

    Parameters:
        raw_candidates: List of raw candidate dicts from EventEngine.
        video_id: Video identifier.
        cooldown_seconds: Cooldown window for merging duplicate proposals (default 10.0 s).
        max_gap_frames: Maximum frame gap for episode continuity (default 60 frames).
        gt_frame: Optional ground truth frame index for window classification.
        temporal_tolerance: Half-window around GT frame (default 45 frames).
        fps: Optional native video frame rate.
        frame_pts_map: Optional authoritative mapping from frame index to native timestamp (seconds).

    Returns:
        List[ConsolidatedEpisode]: Consolidated incident episodes.
    """
    if not raw_candidates:
        return []

    # Dynamic FPS estimation if not provided
    if fps is None:
        for c in raw_candidates:
            cand_fps = c.get("fps", c.get("signals", {}).get("fps"))
            if cand_fps is not None and cand_fps > 0:
                fps = float(cand_fps)
                break
            if c.get("timestamp_s", 0.0) > 0 and c.get("frame", 0) > 0:
                fps = float(c["frame"] / c["timestamp_s"])
                break
        if fps is None:
            fps = 30.0

    # Build fallback frame_pts_map if not provided
    if frame_pts_map is None:
        frame_pts_map = {}
        for c in raw_candidates:
            if "frame" in c and "timestamp_s" in c:
                frame_pts_map[c["frame"]] = float(c["timestamp_s"])

    # Sort chronologically by frame
    sorted_cands = sorted(raw_candidates, key=lambda c: c["frame"])
    episodes: List[ConsolidatedEpisode] = []

    for cand in sorted_cands:
        c_frame = cand["frame"]
        c_time = cand.get("timestamp_s", c_frame / fps)
        c_tracks = set(cand.get("track_ids", []))
        c_saliency = calculate_candidate_saliency(cand, frame_pts_map=frame_pts_map)
        c_type = cand.get("event_type", "")

        matched_ep = None
        for ep in episodes:
            if should_consolidate_into_active_episode(ep, cand, fps=fps):
                matched_ep = ep
                break

        if matched_ep is not None:
            # Aggregate into existing episode
            matched_ep.end_frame = max(matched_ep.end_frame, c_frame)
            matched_ep.end_time_s = max(matched_ep.end_time_s, c_time)
            if cand not in matched_ep.raw_candidates:
                matched_ep.raw_candidates.append(cand)
            for tid in c_tracks:
                if tid not in matched_ep.involved_track_ids:
                    matched_ep.involved_track_ids.append(tid)

            # Prioritize collision_candidate and barrier_strike event type
            if cand.get("event_type") in ("collision_candidate", "barrier_strike"):
                matched_ep.primary_event_type = cand.get("event_type")

            c_tracks_list = sorted(list(c_tracks))
            if matched_ep.primary_actor_pair is None and len(c_tracks_list) == 2 and cand.get("event_type") == "collision_candidate":
                matched_ep.primary_actor_pair = (c_tracks_list[0], c_tracks_list[1])
                c_ground = cand.get("ground_coordinates", (0.0, 0.0))
                if c_ground == (0.0, 0.0):
                    c_ground = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
                if c_ground != (0.0, 0.0):
                    matched_ep.primary_impact_centroid = c_ground
                matched_ep.primary_impact_pixel_centroid = cand.get("signals", {}).get("pixel_centroid", None)
                matched_ep.primary_impact_char_dim = float(cand.get("signals", {}).get("char_dim", 30.0))
            elif matched_ep.primary_actor_pair is not None and len(c_tracks_list) == 2:
                cand_pair = (c_tracks_list[0], c_tracks_list[1])
                if cand_pair != matched_ep.primary_actor_pair and cand_pair not in matched_ep.secondary_actor_pairs:
                    matched_ep.secondary_actor_pairs.append(cand_pair)

            # Update peak candidate if this candidate has higher saliency
            if c_saliency > matched_ep.peak_score:
                matched_ep.peak_score = c_saliency
                matched_ep.peak_candidate = cand
                matched_ep.peak_frame = c_frame
                if len(c_tracks_list) == 2 and (c_tracks_list[0], c_tracks_list[1]) == matched_ep.primary_actor_pair:
                    c_ground = cand.get("ground_coordinates", (0.0, 0.0))
                    if c_ground == (0.0, 0.0):
                        c_ground = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
                    if c_ground != (0.0, 0.0):
                        matched_ep.primary_impact_centroid = c_ground
                    if cand.get("signals", {}).get("pixel_centroid"):
                        matched_ep.primary_impact_pixel_centroid = cand["signals"]["pixel_centroid"]
                    if cand.get("signals", {}).get("char_dim"):
                        matched_ep.primary_impact_char_dim = float(cand["signals"]["char_dim"])
        else:
            # Create a new consolidated episode
            ep_id = f"{video_id}_ep{len(episodes) + 1:02d}_f{c_frame}"
            cand_anom = cand.get("signals", {}).get("anomaly_type", "")
            is_barrier = (cand.get("event_type") in ("barrier_strike", "severe_barrier_impact") or "barrier" in cand_anom)

            c_tracks_list = sorted(list(c_tracks))
            is_collision = cand.get("event_type") == "collision_candidate" or len(c_tracks_list) == 2
            primary_pair = (c_tracks_list[0], c_tracks_list[1]) if is_collision and len(c_tracks_list) == 2 else None
            c_ground = cand.get("ground_coordinates", (0.0, 0.0))
            if c_ground == (0.0, 0.0):
                c_ground = cand.get("signals", {}).get("ground_coordinates", (0.0, 0.0))
            c_px = cand.get("signals", {}).get("pixel_centroid", None)
            c_char_dim = float(cand.get("signals", {}).get("char_dim", 30.0))

            new_ep = ConsolidatedEpisode(
                episode_id=ep_id,
                video_id=video_id,
                start_frame=c_frame,
                end_frame=c_frame,
                start_time_s=c_time,
                end_time_s=c_time,
                primary_event_type=cand.get("event_type", "collision_candidate"),
                involved_track_ids=list(c_tracks),
                raw_candidates=[cand],
                peak_candidate=cand,
                peak_frame=c_frame,
                peak_score=c_saliency,
                initial_trigger_timestamp_s=c_time,
                state="ACTIVE_BARRIER_IMPACT" if is_barrier else "ACTIVE",
                primary_actor_pair=primary_pair,
                secondary_actor_pairs=[],
                primary_impact_centroid=c_ground if c_ground != (0.0, 0.0) else None,
                primary_impact_pixel_centroid=c_px,
                primary_impact_char_dim=c_char_dim,
            )
            episodes.append(new_ep)

    # Classify in-window vs out-of-window relative to ground truth
    if gt_frame is not None:
        for ep in episodes:
            in_window = (
                abs(ep.peak_frame - gt_frame) <= temporal_tolerance
                or (ep.start_frame <= gt_frame <= ep.end_frame)
                or (abs(ep.start_frame - gt_frame) <= temporal_tolerance)
                or (abs(ep.end_frame - gt_frame) <= temporal_tolerance)
            )
            ep.is_in_gt_window = in_window

    return episodes
