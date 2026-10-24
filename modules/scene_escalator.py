from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np


from enum import Enum


class SceneEscalatorDisposition(str, Enum):
    ESCALATED_PHYSICAL_INCIDENT = "ESCALATED_PHYSICAL_INCIDENT"
    SUPPRESSED_BENIGN_AFTERMATH = "SUPPRESSED_BENIGN_AFTERMATH"
    FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW = "FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW"


class EpisodeStatus(str, Enum):
    ACTIVE = "ACTIVE"
    CONFIRMED = "CONFIRMED"
    SUPPRESSED_BENIGN_AFTERMATH = "SUPPRESSED_BENIGN_AFTERMATH"


@dataclass
class StreamMetadata:
    fps: float
    duration_sec: float
    total_frames: int
    width: int
    height: int


@dataclass
class TrackState:
    frame_idx: int
    pts_ns: int
    bbox: Tuple[float, float, float, float]
    ground_pos: Tuple[float, float]
    speed_kmh: float
    heading_deg: float
    accel_ms2: float = 0.0
    class_label: str = 'vehicle'
    track_age: int = 0
    status: str = 'active'


@dataclass
class CandidateProposal:
    frame_idx: int
    event_type: str
    track_ids: List[int]
    video_id: str = ''
    confidence: float = 0.5
    signals: Dict[str, Any] = field(default_factory=dict)
    description: str = ''
    timestamp_sec: Optional[float] = None

    def __post_init__(self):
        if self.timestamp_sec is None:
            self.timestamp_sec = float(
                self.signals.get("timestamp_s", self.signals.get("timestamp_sec", self.frame_idx / 30.0))
            )

    @classmethod
    def from_dict(cls, d: Dict[str, Any], video_id: str = '') -> 'CandidateProposal':
        f = d.get('frame', d.get('frame_idx', 0))
        t_sec = float(d.get('timestamp_s', d.get('timestamp_sec', f / 30.0)))
        return cls(
            frame_idx=f,
            event_type=d.get('event_type', ''),
            track_ids=list(d.get('track_ids', [])),
            video_id=d.get('video_id', video_id),
            confidence=float(d.get('confidence', 0.5)),
            signals=dict(d.get('signals', {})),
            description=d.get('description', ''),
            timestamp_sec=t_sec,
        )


@dataclass
class SceneEscalationResult:
    disposition: SceneEscalatorDisposition = SceneEscalatorDisposition.SUPPRESSED_BENIGN_AFTERMATH
    score: Optional[float] = None
    window_state: str = "COMPLETE_AFTERMATH_WINDOW"
    canonical_type: str = "Unknown"
    telemetry: Dict[str, Any] = field(default_factory=dict)
    channels: Dict[str, float] = field(default_factory=dict)
    safeguards: Dict[str, Any] = field(default_factory=dict)
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disposition": self.disposition.value if hasattr(self.disposition, "value") else str(self.disposition),
            "score": self.score,
            "window_state": self.window_state,
            "canonical_type": self.canonical_type,
            "telemetry": self.telemetry,
            "channels": self.channels,
            "safeguards": self.safeguards,
            "escalated": self.escalated,
        }


class TrackerContext:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.history_by_tid: Dict[int, List[TrackState]] = {}
        self.frame_to_time: Dict[int, float] = {}

    def record_frame(
        self,
        frame_idx: int,
        pts_ns: int,
        timestamp_s: float,
        active_tracks: Dict[int, Any],
        kinematics: Dict[int, Any],
    ) -> None:
        self.frame_to_time[frame_idx] = timestamp_s
        for tid, tr in active_tracks.items():
            k = kinematics.get(tid)
            if k is not None:
                ground_pos = tuple(k.ground_pos) if hasattr(k, 'ground_pos') and k.ground_pos is not None else (0.0, 0.0)
                speed_kmh = float(k.speed_kmh) if hasattr(k, 'speed_kmh') and k.speed_kmh is not None else 0.0
                heading_deg = float(k.heading_deg) if hasattr(k, 'heading_deg') and k.heading_deg is not None else 0.0
                accel_ms2 = float(k.acceleration_mag) if hasattr(k, 'acceleration_mag') and k.acceleration_mag is not None else 0.0
            else:
                ground_pos = tr.ground_history[-1] if getattr(tr, 'ground_history', None) else (0.0, 0.0)
                speed_kmh = float(getattr(tr, 'max_recent_speed_kmh', 0.0))
                heading_deg = 0.0
                accel_ms2 = 0.0

            bbox = tuple(tr.current_bbox_xywh) if hasattr(tr, 'current_bbox_xywh') else (0.0, 0.0, 0.0, 0.0)
            st = TrackState(
                frame_idx=frame_idx,
                pts_ns=pts_ns,
                bbox=bbox,
                ground_pos=ground_pos,
                speed_kmh=speed_kmh,
                heading_deg=heading_deg,
                accel_ms2=accel_ms2,
                class_label=getattr(tr, 'class_label', 'vehicle'),
                track_age=getattr(tr, 'track_age', 0),
                status=getattr(tr, 'status', 'active'),
            )
            if tid not in self.history_by_tid:
                self.history_by_tid[tid] = []
            self.history_by_tid[tid].append(st)

    def get_history_window(self, t_start: float, t_end: float) -> Dict[int, List[TrackState]]:
        dt_buffer = 0.05
        t_min = t_start - dt_buffer
        t_max = t_end + dt_buffer
        res: Dict[int, List[TrackState]] = {}
        for tid, states in self.history_by_tid.items():
            matching = [s for s in states if t_min <= (s.pts_ns * 1e-9) <= t_max]
            if matching:
                res[tid] = matching
        return res


class SceneEscalator:
    WEIGHTS: Dict[str, Dict[str, float]] = {
        'Type 1': {'w_v': 0.30, 'w_rest': 0.15, 'w_rot': 0.10, 'w_ovl': 0.15, 'w_lat': 0.15, 'w_push': 0.05, 'w_up': 0.10},
        'Type 2': {'w_v': 0.15, 'w_rest': 0.30, 'w_rot': 0.05, 'w_ovl': 0.20, 'w_lat': 0.15, 'w_push': 0.05, 'w_up': 0.10},
        'Type 3': {'w_v': 0.20, 'w_rest': 0.15, 'w_rot': 0.30, 'w_ovl': 0.10, 'w_lat': 0.10, 'w_push': 0.10, 'w_up': 0.05},
        'Type 4': {'w_v': 0.15, 'w_rest': 0.10, 'w_rot': 0.15, 'w_ovl': 0.20, 'w_lat': 0.30, 'w_push': 0.05, 'w_up': 0.05},
        'Type 5': {'w_v': 0.30, 'w_rest': 0.30, 'w_rot': 0.25, 'w_ovl': 0.00, 'w_lat': 0.00, 'w_push': 0.00, 'w_up': 0.15},
    }

    TYPE_NAMES: Dict[str, str] = {
        'Type 1': 'High-Speed Highway Rear-End',
        'Type 2': 'Low-Speed Urban Rear-End',
        'Type 3': 'T-Bone / Intersection Angle',
        'Type 4': 'Sideswipe / Cut-in',
        'Type 5': 'Single-Vehicle Barrier / Spin',
    }

    def __init__(self, tau_aftermath: float = 0.50, k_upstream: float = 3.0):
        self.tau = tau_aftermath
        self.k_up = k_upstream

    @staticmethod
    def _compute_bbox_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        ax1, ay1, aw, ah = boxA
        bx1, by1, bw, bh = boxB
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        areaA = max(0.0, aw * ah)
        areaB = max(0.0, bw * bh)
        union = areaA + areaB - intersection
        if union <= 1e-6:
            return 0.0
        return float(intersection / union)

    def classify_topology(
        self,
        candidate: CandidateProposal,
        track_history: Dict[int, List[TrackState]],
        v_entry: float
    ) -> str:
        ctype_sig = candidate.signals.get('canonical_type')
        if ctype_sig in self.WEIGHTS:
            return ctype_sig

        etype = candidate.event_type.lower()
        tids = candidate.track_ids

        if len(tids) == 1 or etype in ['single_vehicle_anomaly', 'barrier_strike', 'spin_out', 'rollover']:
            return 'Type 5'

        sig = candidate.signals
        is_tbone = sig.get('is_tbone', False)
        hdg_delta = sig.get('heading_delta_deg', sig.get('heading_delta', 0.0))
        is_crossing = sig.get('is_crossing_trajectory', False)
        iou = sig.get('bbox_iou', 0.0)

        if is_tbone or is_crossing or hdg_delta > 60.0:
            return 'Type 3'

        if 20.0 <= hdg_delta <= 60.0 or iou >= 0.40:
            return 'Type 4'

        if v_entry <= 35.0:
            return 'Type 2'
        else:
            return 'Type 1'

    def evaluate_candidate(
        self,
        candidate: Union[CandidateProposal, Dict[str, Any]],
        track_history: Dict[int, List[TrackState]],
        fps: float = 30.0,
        horizon_seconds: float = 1.5,
    ) -> SceneEscalationResult:
        if isinstance(candidate, dict):
            candidate = CandidateProposal.from_dict(candidate)
        f0 = candidate.frame_idx
        tids = candidate.track_ids
        horizon_frames = int(round(horizon_seconds * fps))
        f_end = f0 + horizon_frames

        t_hist_by_tid: Dict[int, Dict[int, TrackState]] = {}
        for tid, states in track_history.items():
            t_hist_by_tid[tid] = {s.frame_idx: s for s in states}

        p_tids = tids[:2] if len(tids) >= 2 else (tids[:1] if len(tids) == 1 else [])
        
        entry_speeds: List[float] = []
        for tid in p_tids:
            st = t_hist_by_tid.get(tid, {}).get(f0)
            if st:
                entry_speeds.append(st.speed_kmh)
            else:
                pre_f = [f for f in t_hist_by_tid.get(tid, {}).keys() if f <= f0]
                if pre_f:
                    entry_speeds.append(t_hist_by_tid[tid][max(pre_f)].speed_kmh)

        if not entry_speeds:
            v_entry = float(candidate.signals.get('v_entry', candidate.signals.get('v_entry_kmh', candidate.signals.get('speed_kmh', 30.0))))
        else:
            v_entry = max(entry_speeds)

        ctype = self.classify_topology(candidate, track_history, v_entry)
        weights = self.WEIGHTS[ctype]

        post_frames = list(range(f0, f_end + 1))

        has_replay_telemetry = any(k in candidate.signals for k in ['dv_sust', 'tau_rest', 'delta_theta'])
        if has_replay_telemetry:
            dv_sust = float(candidate.signals.get('dv_sust', 0.0))
            tau_rest = float(candidate.signals.get('tau_rest', 0.0))
            dtheta_post = float(candidate.signals.get('delta_theta', 0.0))
            tau_overlap = float(candidate.signals.get('tau_overlap', 0.0))
            delta_d_lat = float(candidate.signals.get('delta_d_lat', 0.0))
            push_disp = float(candidate.signals.get('push_disp', 0.0))
            upstream_react = int(candidate.signals.get('upstream_react', 0))
        else:
            sust_speeds = []
            for tid in p_tids:
                speeds = [t_hist_by_tid[tid][f].speed_kmh for f in post_frames if f in t_hist_by_tid.get(tid, {})]
                if speeds:
                    sust_speeds.append(sorted(speeds)[len(speeds) // 2])
            
            v_post_sust = min(sust_speeds) if sust_speeds else v_entry
            dv_sust = max(0.0, v_entry - v_post_sust)

            rest_counts = []
            for tid in p_tids:
                cnt = sum(1 for f in post_frames if f in t_hist_by_tid.get(tid, {}) and t_hist_by_tid[tid][f].speed_kmh <= 5.0)
                rest_counts.append(cnt)
            tau_rest = (max(rest_counts) / fps) if rest_counts else 0.0

            hdg_deltas = []
            for tid in p_tids:
                hdgs = [t_hist_by_tid[tid][f].heading_deg for f in post_frames if f in t_hist_by_tid.get(tid, {})]
                if hdgs:
                    h0 = hdgs[0]
                    diffs = [abs(h - h0) for h in hdgs]
                    norm_diffs = [d if d <= 180.0 else 360.0 - d for d in diffs]
                    hdg_deltas.append(max(norm_diffs))
            dtheta_post = max(hdg_deltas) if hdg_deltas else 0.0

            if len(p_tids) >= 2:
                t1, t2 = p_tids[0], p_tids[1]
                ovl_frames = 0
                for f in post_frames:
                    if f in t_hist_by_tid.get(t1, {}) and f in t_hist_by_tid.get(t2, {}):
                        b1 = t_hist_by_tid[t1][f].bbox
                        b2 = t_hist_by_tid[t2][f].bbox
                        if self._compute_bbox_iou(b1, b2) >= 0.05:
                            ovl_frames += 1
                tau_overlap = ovl_frames / fps
            else:
                tau_overlap = 0.0

            if len(p_tids) >= 2:
                t1, t2 = p_tids[0], p_tids[1]
                g_dists = []
                for f in post_frames:
                    if f in t_hist_by_tid.get(t1, {}) and f in t_hist_by_tid.get(t2, {}):
                        g1 = t_hist_by_tid[t1][f].ground_pos
                        g2 = t_hist_by_tid[t2][f].ground_pos
                        g_dists.append(math.hypot(g1[0] - g2[0], g1[1] - g2[1]))
                delta_d_lat = abs(g_dists[-1] - g_dists[0]) if len(g_dists) >= 2 else 0.0
            else:
                delta_d_lat = 0.0

            push_disps = []
            for tid in p_tids:
                pts = [t_hist_by_tid[tid][f].ground_pos for f in post_frames if f in t_hist_by_tid.get(tid, {})]
                if len(pts) >= 2:
                    push_disps.append(math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
            push_disp = max(push_disps) if push_disps else 0.0

            upstream_react = 0
            for tid, hist in t_hist_by_tid.items():
                if tid in p_tids:
                    continue
                bg_spds = [hist[f].speed_kmh for f in post_frames if f in hist]
                if len(bg_spds) >= 5:
                    if (bg_spds[0] - min(bg_spds) >= 15.0) or (min(bg_spds) <= 5.0):
                        upstream_react += 1

        dv_denom = max(15.0, 0.6 * max(0.0, v_entry))
        phi_dv = min(1.0, max(0.0, dv_sust) / dv_denom)

        crawl_guard_active = bool(v_entry <= 8.0)
        crawl_override_passed = bool(push_disp > 0.50 or tau_overlap >= 0.30)
        if crawl_guard_active and not crawl_override_passed:
            phi_rest = 0.0
        else:
            phi_rest = min(1.0, max(0.0, tau_rest) / 1.0)

        phi_rot = min(1.0, max(0.0, dtheta_post) / 60.0)
        phi_ovl = min(1.0, max(0.0, tau_overlap) / 0.50)
        phi_lat = min(1.0, max(0.0, delta_d_lat) / 2.00)
        phi_push = min(1.0, max(0.0, push_disp) / 2.00)
        phi_up = min(1.0, max(0.0, float(upstream_react)) / getattr(self, "k_up", 3.0))

        e_aftermath = (
            weights['w_v'] * phi_dv +
            weights['w_rest'] * phi_rest +
            weights['w_rot'] * phi_rot +
            weights['w_ovl'] * phi_ovl +
            weights['w_lat'] * phi_lat +
            weights['w_push'] * phi_push +
            weights['w_up'] * phi_up
        )
        score = round(float(e_aftermath), 4)
        escalated = bool(score >= self.tau)

        channels = {
            'phi_dv': round(phi_dv, 4),
            'phi_rest': round(phi_rest, 4),
            'phi_rot': round(phi_rot, 4),
            'phi_ovl': round(phi_ovl, 4),
            'phi_lat': round(phi_lat, 4),
            'phi_push': round(phi_push, 4),
            'phi_up': round(phi_up, 4),
        }

        telemetry = {
            'v_entry': round(v_entry, 2),
            'dv_sust': round(dv_sust, 2),
            'tau_rest': round(tau_rest, 2),
            'delta_theta': round(dtheta_post, 2),
            'tau_overlap': round(tau_overlap, 2),
            'delta_d_lat': round(delta_d_lat, 2),
            'push_disp': round(push_disp, 2),
            'upstream_react': int(upstream_react),
        }

        safeguards = {
            'dv_denominator': round(dv_denom, 2),
            'crawl_guard_active': crawl_guard_active,
            'crawl_override_passed': crawl_override_passed,
        }

        # Physical Consequence Safeguard: Rear-end collisions (Type 1 & Type 2) require real kinetic impact
        # (either sustained velocity drop dv_sust >= 4.0 km/h or kinetic arrest tau_rest >= 0.25 s).
        # Cruising vehicles with dv_sust < 4.0 and tau_rest < 0.25 are passing/adjacent-lane flow.
        if ctype in ('Type 1', 'Type 2') and dv_sust < 4.0 and tau_rest < 0.25:
            escalated = False

        # Kinetic Aftermath Rest Safeguard:
        # In physics, severe crashes lead to kinetic energy dissipation and sustained vehicle rest
        # (speed <= 5.0 km/h for >= 0.25s). Passing / cruising traffic where vehicles maintain continuous
        # speed without any rest (tau_rest < 0.25 s) is suppressed as benign roadway flow, unless an explicit
        # post_event_rest signal is asserted.
        has_post_rest_signal = bool(candidate.signals.get('post_event_rest', False))
        if tau_rest < 0.25 and not has_post_rest_signal:
            escalated = False

        disp = (
            SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT
            if escalated
            else SceneEscalatorDisposition.SUPPRESSED_BENIGN_AFTERMATH
        )

        return SceneEscalationResult(
            disposition=disp,
            score=score,
            window_state="COMPLETE_AFTERMATH_WINDOW",
            canonical_type=ctype,
            telemetry=telemetry,
            channels=channels,
            safeguards=safeguards,
            escalated=escalated,
        )
