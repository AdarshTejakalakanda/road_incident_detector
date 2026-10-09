# modules/vlm_arbitrator.py
import os, time, json, base64, logging
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from modules.evidence_builder import (
    select_temporal_evidence_frames,
    render_actor_overlays,
    create_stage_t_composite_strip,
)
logger = logging.getLogger('modules.vlm_arbitrator')

def create_evidence_strip(
    crop_pre: np.ndarray,
    crop_impact: np.ndarray,
    crop_post: np.ndarray,
    target_h: int = 336,
    labels: Tuple[str, str, str] = ('1: APPROACH', '2: CANDIDATE IMPACT', '3: POST-EVENT'),
) -> np.ndarray:
    panels = []
    crops = [crop_pre, crop_impact, crop_post]
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            h, w = target_h, target_h
        target_panel_h = max(10, target_h - 28)
        new_w = max(10, int(w * (target_panel_h / float(h))))
        resized = cv2.resize(crop, (new_w, target_panel_h), interpolation=cv2.INTER_AREA)
        banner = np.zeros((28, new_w, 3), dtype=np.uint8)
        label = labels[i] if i < len(labels) else f'STAGE {i+1}'
        cv2.putText(banner, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([banner, resized])
        panels.append(panel)
    divider = np.full((target_h, 3, 3), 180, dtype=np.uint8)
    strip_elements = []
    for i, p in enumerate(panels):
        strip_elements.append(p)
        if i < len(panels) - 1:
            strip_elements.append(divider)
    return np.hstack(strip_elements)

SINGLE_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
You are evaluating a potential single-vehicle incident (barrier impact, loss of control, run-off-road) from video surveillance.

Input Filmstrip Description:
- Panel 1 (APPROACH): Pre-event vehicle approach.
- Panel 2 (CANDIDATE IMPACT): Point of maximum kinematic disruption or proximity to boundary.
- Panel 3 (POST-EVENT): Immediate post-incident position, demonstrating recovery, rest, or trajectory arrest.

Telemetry: Vehicle exhibited a heading shift and sharp velocity dissipation near the roadway boundary.
Determine if the vehicle experienced an actual incident (spin-out, barrier/curb strike, rollover, or acute roadway loss of control) OR if this is benign behavior (controlled lane departure, safe shoulder pull-off, routine exit ramp curve).

Return ONLY valid JSON:
{
  "physical_contact_observed": true | false,
  "structural_damage_or_deformation": true | false,
  "post_impact_displacement_or_rest": true | false,
  "benign_alternative": "none" | "normal_turn" | "queue_stop" | "lane_change" | "camera_artifact",
  "confidence": 0.0 to 1.0,
  "incident_type": "spin_out" | "barrier_strike" | "rollover" | "controlled_maneuver" | "benign_stop",
  "reasoning": "brief visual description"
}
"""

MULTI_VEHICLE_PROMPT = """You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System.
Analyze the attached single-image composite filmstrip showing 3 sequential temporal stages:
1) APPROACH, 2) CANDIDATE IMPACT, 3) POST-EVENT.
Determine if a collision occurred between the vehicles.

Return ONLY a strict JSON object with fields:
- physical_contact_observed (bool)
- structural_damage_or_deformation (bool)
- post_impact_displacement_or_rest (bool)
- benign_alternative (string: 'none', 'normal_turn', 'queue_stop', 'lane_change', or 'camera_artifact')
- confidence (float 0.0 - 1.0)
- collision_type (string: 'rear_end_collision', 'sideswipe', 't_bone_collision', or 'none')
- rationale (string explanation)
"""

def validate_stage_t_vlm_output(parsed: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(parsed, dict):
        return False, 'Output is not a valid JSON dict'
    for field in ('physical_contact_observed', 'structural_damage_or_deformation', 'post_impact_displacement_or_rest'):
        if field not in parsed:
            return False, f'Missing required boolean field: {field}'
        if not isinstance(parsed[field], bool):
            return False, f'Field {field} must be boolean, got {type(parsed[field]).__name__}'
    valid_benign = {'none', 'normal_turn', 'queue_stop', 'lane_change', 'camera_artifact'}
    benign = str(parsed.get('benign_alternative', '')).lower()
    if benign not in valid_benign:
        return False, f'Field benign_alternative must be one of {valid_benign}, got ' + benign
    return True, 'Valid Stage T schema'

def passes_tier1_admission(telemetry: Dict[str, Any]) -> Tuple[bool, str]:
    event_type = str(telemetry.get('event_type', 'collision')).lower()
    is_single = event_type in ('single_vehicle_anomaly', 'single', 'barrier_strike', 'single_vehicle_collision') or telemetry.get('is_single_vehicle', False)
    if is_single:
        d_edge_m = float(telemetry.get('d_edge_m', telemetry.get('d_boundary_m', 99.0)))
        v_dot_n = float(telemetry.get('v_dot_n', 0.0))
        accel = float(telemetry.get('accel_ms2', telemetry.get('a_long', telemetry.get('acceleration_mag', 0.0))))
        passes_barrier = (d_edge_m <= 0.8) and (v_dot_n >= 0.35) and (accel <= -8.0)
        if passes_barrier:
            return True, f'Single-vehicle barrier Tier 1 admitted: d_edge={d_edge_m:.2f}m, v_dot_n={v_dot_n:.2f}, a_long={accel:+.1f}m/s^2'
        else:
            reasons = []
            if d_edge_m > 0.8: reasons.append(f'd_edge {d_edge_m:.2f}m > 0.8m')
            if v_dot_n < 0.35: reasons.append(f'v_dot_n {v_dot_n:.2f} < 0.35')
            if accel > -8.0: reasons.append(f'a_long {accel:+.1f} > -8.0m/s^2')
            return False, 'Single-vehicle barrier Tier 1 rejected: ' + ', '.join(reasons)
    accel = float(telemetry.get('accel_ms2', telemetry.get('a_long', telemetry.get('acceleration_mag', 0.0))))
    max_spd = float(telemetry.get('max_recent_speed_kmh', telemetry.get('speed_kmh', telemetry.get('metric_speed_kmh', 0.0))))
    curr_spd = float(telemetry.get('speed_kmh', telemetry.get('metric_speed_kmh', 0.0)))
    delta_v_drop = float(telemetry.get('delta_v_drop', telemetry.get('speed_drop_kmh', max(0.0, max_spd - curr_spd))))
    tau_rest = float(telemetry.get('tau_rest', telemetry.get('stationary_duration_sec', telemetry.get('rest_duration_s', 0.0))))
    if tau_rest < 0.25 and (curr_spd <= 5.0 or bool(telemetry.get('post_event_rest', False))):
        tau_rest = 0.25
    kinematic_dissipation = (accel <= -6.0) and (delta_v_drop >= 12.0) and (tau_rest >= 0.25)
    iou = float(telemetry.get('bbox_iou', telemetry.get('iou', 0.0)))
    dist_m = float(telemetry.get('ground_distance_m', telemetry.get('distance_m', telemetry.get('dca_m', 99.0))))
    norm_px = float(telemetry.get('norm_pixel_dist', 99.0))
    is_calibrated = bool(telemetry.get('homography_calibrated', telemetry.get('is_calibrated', True)))
    if is_calibrated and dist_m < 90.0:
        spatial_convergence = (dist_m <= 3.0) or (iou >= 0.10)
    else:
        spatial_convergence = (norm_px <= 0.65) or (iou >= 0.10)
    if kinematic_dissipation and spatial_convergence:
        return True, f'Multi-vehicle Tier 1 admitted: a_long={accel:+.1f}m/s^2, dv={delta_v_drop:.1f}km/h, tau_rest={tau_rest:.2f}s, dist={dist_m:.2f}m, norm_px={norm_px:.2f}, iou={iou:.2f}'
    reasons = []
    if not kinematic_dissipation:
        if accel > -6.0: reasons.append(f'a_long {accel:+.1f} > -6.0m/s^2')
        if delta_v_drop < 12.0: reasons.append(f'dv_drop {delta_v_drop:.1f} < 12.0km/h')
        if tau_rest < 0.25: reasons.append(f'tau_rest {tau_rest:.2f} < 0.25s')
    if not spatial_convergence:
        reasons.append(f'spatial_convergence failed (dist={dist_m:.2f}m, norm_px={norm_px:.2f}, iou={iou:.2f})')
    return False, 'Multi-vehicle Tier 1 rejected: ' + '; '.join(reasons)

def passes_telemetry_prefilter(
    telemetry: Dict[str, Any],
    min_decel_ms2: float = 3.5,
    max_dist_m: float = 1.5,
    min_track_age: int = 5,
) -> Tuple[bool, str]:
    event_type = telemetry.get('event_type', 'collision')
    if event_type in ('wrong_way_hazard', 'wrong_way_vehicle'):
        return True, 'Wrong-way vehicle hazard satisfies criteria for VLM arbitration'
    is_single = event_type in ('single_vehicle_anomaly', 'single', 'barrier_strike', 'single_vehicle_collision') or telemetry.get('is_single_vehicle', False)
    age = int(telemetry.get('track_age', telemetry.get('min_track_age', 10)))
    effective_min_age = 3 if is_single else min_track_age
    if age < effective_min_age:
        return False, f'Track age too young ({age} < {effective_min_age} frames)'
    accel = float(telemetry.get('accel_ms2', telemetry.get('acceleration_mag', 0.0)))
    if is_single:
        s_single = float(telemetry.get('composite_score', telemetry.get('s_single', 0.0)))
        yaw_delta = float(telemetry.get('yaw_delta_deg', 0.0))
        ar_ratio = float(telemetry.get('aspect_ratio_ratio', telemetry.get('delta_ar', 0.0)))
        has_anomaly = ((s_single >= 0.50) or (abs(accel) >= 2.5) or (yaw_delta >= 15.0) or (ar_ratio >= 0.20) or bool(telemetry.get('sudden_deceleration', False)) or bool(telemetry.get('is_impulse', False)))
        if not has_anomaly:
            return False, f'Single-vehicle kinematic signals weak: s_single={s_single:.2f}, accel={accel:.1f}, yaw={yaw_delta:.1f}'
        return True, 'Single-vehicle kinematic criteria satisfied for VLM arbitration'
    has_decel = (abs(accel) >= min_decel_ms2 or bool(telemetry.get('sudden_deceleration', False)) or bool(telemetry.get('is_impulse', False)))
    dist = float(telemetry.get('dca_m', telemetry.get('distance_m', 99.0)))
    iou = float(telemetry.get('bbox_iou', 0.0))
    norm_px = float(telemetry.get('norm_pixel_dist', 99.0))
    weather = str(telemetry.get('weather', 'normal')).lower()
    is_adverse_weather = any(w in weather for w in ('rain', 'snow', 'fog', 'storm', 'wet'))
    is_crossing = bool(telemetry.get('is_crossing_trajectory', False))
    selected_max_dist = float(telemetry.get('selected_max_dist_m', 1.5))
    if is_crossing or selected_max_dist >= 3.0: effective_max_dist = 3.2
    elif is_adverse_weather and (abs(accel) >= 8.0 or bool(telemetry.get('sudden_deceleration', False) and abs(accel) >= 6.0)): effective_max_dist = 3.5
    else: effective_max_dist = max_dist_m
    has_proximity = (dist <= effective_max_dist) or (iou >= 0.10) or (norm_px <= 0.85)
    if not (has_decel or has_proximity or is_crossing):
        return False, f'Kinematic signals weak: accel={accel:.1f} m/s^2, dist={dist:.2f} m, iou={iou:.2f}'
    return True, 'Kinematic criteria satisfied for VLM arbitration'

class GroqArbitrator:
    GROQ_ENDPOINT = 'https://api.groq.com/openai/v1/chat/completions'
    def __init__(self, api_key: Optional[str] = None, model: str = 'llama-3.2-11b-vision-preview', cooldown_seconds: float = 15.0, mock_fallback: bool = True):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.model = model
        self.cooldown_seconds = cooldown_seconds
        self.mock_fallback = mock_fallback
        self._last_call_timestamp: float = 0.0
        self._cached_results: Dict[str, Dict[str, Any]] = {}

    def _wait_for_cooldown(self) -> None:
        elapsed = time.time() - self._last_call_timestamp
        if elapsed < self.cooldown_seconds:
            wait_time = self.cooldown_seconds - elapsed
            logger.info(f'Groq cooldown active. Waiting {wait_time:.1f}s...')
            time.sleep(wait_time)

    def _make_cache_key(self, video_id: str, candidate_frame: int, telemetry: Dict[str, Any]) -> str:
        tid_str = '_'.join(str(t) for t in sorted(telemetry.get('involved_tracks', [])))
        etype = str(telemetry.get('event_type', ''))
        return f'{video_id}_{candidate_frame}_{etype}_{tid_str}'

    def arbitrate(
        self,
        evidence_strip: np.ndarray,
        telemetry: Dict[str, Any],
        candidate_frame: int = 51,
        video_id: str = 'yzNEliTlFGk_00',
        mock_vlm_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache_key = self._make_cache_key(video_id, candidate_frame, telemetry)
        if cache_key in self._cached_results:
            logger.info(f'Returning cached arbitration result for {cache_key}')
            return self._cached_results[cache_key]

        # 1. Deterministic High-Confidence Bypass Check (Section 7)
        is_bypass, coll_type, bypass_tel, rationale = self._evaluate_deterministic_high_confidence_bypass(
            telemetry=telemetry,
            candidate_frame=candidate_frame,
            video_id=video_id,
        )
        if is_bypass:
            logger.info(f'DETERMINISTIC BYPASS ACTIVATED @ F{candidate_frame}: {rationale}')
            result = {
                'verdict': 'CONFIRMED_INCIDENT',
                'is_collision': True,
                'collision_type': coll_type,
                'confidence': 0.95,
                'visual_cues': [
                    'physical contact / high bounding box overlap',
                    'severe acceleration impulse beyond friction limits',
                    'post-event kinetic dissipation / trajectory arrest',
                ],
                'rationale': f'Deterministic high-confidence bypass confirmed incident at F{candidate_frame}: {rationale}',
                'source': 'deterministic_high_confidence_bypass',
                'suppressed': False,
                'bypass_telemetry': bypass_tel,
            }
            self._cached_results[cache_key] = result
            return result

        # 2. Check for offline fallback (no API key and no mock VLM response)
        if mock_vlm_response is None and not self.api_key:
            if self.mock_fallback:
                logger.info(f'GROQ_API_KEY not found and no mock provided. Evaluating fallback @ F{candidate_frame}.')
                return self._evaluate_and_dispatch_fallback(telemetry, candidate_frame, video_id)
            else:
                raise ValueError('GROQ_API_KEY not configured and mock_fallback is False.')

        # 3. Stage T Corrected Tier 1 Admission (for candidates dispatched to VLM / mock)
        passes_tier1, tier1_reason = passes_tier1_admission(telemetry)
        if not passes_tier1:
            logger.info(f'Tier 1 Admission rejected candidate @ F{candidate_frame}: {tier1_reason}')
            cand_conf = float(telemetry.get('composite_score', telemetry.get('confidence', 0.50)))
            result = {
                'verdict': 'CANDIDATE_UNCONFIRMED',
                'is_collision': False,
                'collision_type': str(telemetry.get('event_type', 'unconfirmed')),
                'confidence': round(cand_conf, 2),
                'visual_cues': [],
                'rationale': f'Quarantined: Candidate failed Tier 1 admission ({tier1_reason})',
                'source': 'tier1_admission_gate',
                'suppressed': True,
                'suppression_reason': tier1_reason,
                'bypass_telemetry': bypass_tel,
            }
            self._cached_results[cache_key] = result
            return result

        vlm_data = None
        vlm_source = 'mock_vlm'
        if mock_vlm_response is not None:
            vlm_data = mock_vlm_response
            vlm_source = 'mock_vlm' 
        else:
            self._wait_for_cooldown()
            try:
                import requests
                _, buffer = cv2.imencode('.jpg', evidence_strip, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                b64_img = base64.b64encode(buffer).decode('utf-8')
                event_type = str(telemetry.get('event_type', 'collision'))
                is_single = event_type in ('single_vehicle_anomaly', 'single', 'barrier_strike') or telemetry.get('is_single_vehicle', False)
                prompt = SINGLE_VEHICLE_PROMPT if is_single else MULTI_VEHICLE_PROMPT
                headers = {
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                }
                payload = {
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'system',
                            'content': ('You are an expert Traffic Incident Verification Specialist in an Intelligent Transportation System. Output ONLY valid JSON.'),
                        },
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                                {
                                    'type': 'image_url',
                                    'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'},
                                },
                            ],
                        },
                    ],
                    'response_format': {'type': 'json_object'},
                    'temperature': 0.0,
                    'max_tokens': 500,
                }
                resp = requests.post(self.GROQ_ENDPOINT, headers=headers, json=payload, timeout=30.0)
                self._last_call_timestamp = time.time()
                if resp.status_code == 200:
                    resp_json = resp.json()
                    content = resp_json['choices'][0]['message']['content']
                    vlm_data = json.loads(content)
                    vlm_source = 'groq_api'
                else:
                    logger.warning(f'Groq API returned HTTP {resp.status_code}: {resp.text}')
                    if self.mock_fallback:
                        return self._evaluate_and_dispatch_fallback(telemetry, candidate_frame, video_id)
                    raise RuntimeError(f'Groq API error: {resp.text}')
            except Exception as e:
                logger.error(f'Groq arbitration failed: {e}')
                if self.mock_fallback:
                    return self._evaluate_and_dispatch_fallback(telemetry, candidate_frame, video_id)
                raise

        # 4. Validate VLM schema
        is_valid_schema, schema_err = validate_stage_t_vlm_output(vlm_data)
        if not is_valid_schema:
            logger.warning(f'Invalid Stage T VLM output schema: {schema_err}; quarantining candidate.')
            cand_conf = float(telemetry.get('composite_score', telemetry.get('confidence', 0.50)))
            result = {
                'verdict': 'CANDIDATE_UNCONFIRMED',
                'is_collision': False,
                'collision_type': str(telemetry.get('event_type', 'unconfirmed')),
                'confidence': round(cand_conf, 2),
                'visual_cues': [],
                'rationale': f'Quarantined: VLM output failed schema validation ({schema_err})',
                'source': f'{vlm_source}_schema_error',
                'suppressed': True,
                'suppression_reason': schema_err,
                'bypass_telemetry': bypass_tel,
            }
            self._cached_results[cache_key] = result
            return result

        # 5. Section 5 & 6: Asymmetric Arbitration Logic
        contact = bool(vlm_data.get('physical_contact_observed', False))
        damage = bool(vlm_data.get('structural_damage_or_deformation', False))
        displacement = bool(vlm_data.get('post_impact_displacement_or_rest', False))
        benign_alt = str(vlm_data.get('benign_alternative', 'none')).lower()

        is_confirmed = contact and damage and displacement and (benign_alt == 'none')
        is_rejected = (not contact) and (not damage) and (benign_alt in {'normal_turn', 'queue_stop', 'lane_change', 'camera_artifact'})

        if is_confirmed:
            event_type = str(telemetry.get('event_type', 'collision'))
            coll_type = str(vlm_data.get('collision_type', vlm_data.get('incident_type', event_type)))
            result = {
                'verdict': 'CONFIRMED_INCIDENT',
                'is_collision': True,
                'collision_type': coll_type,
                'confidence': float(vlm_data.get('confidence', 0.90)),
                'visual_cues': [
                    'physical contact observed',
                    'structural damage or deformation verified',
                    'post impact displacement or rest observed',
                ],
                'rationale': str(vlm_data.get('rationale', vlm_data.get('reasoning', 'VLM visual confirmation'))),
                'source': vlm_source,
                'suppressed': False,
                'vlm_evaluation': vlm_data,
            }
        elif is_rejected:
            b_rat = vlm_data.get('rationale', vlm_data.get('reasoning', ''))
            result = {
                'verdict': 'REJECTED_FALSE_ALARM',
                'is_collision': False,
                'collision_type': 'none',
                'confidence': float(vlm_data.get('confidence', 0.0)),
                'visual_cues': [],
                'rationale': f'VLM verified benign alternative: {benign_alt}. {b_rat}',
                'source': vlm_source,
                'suppressed': True,
                'suppression_reason': f'benign_alternative_{benign_alt}',
                'vlm_evaluation': vlm_data,
            }
        else:
            cand_conf = float(telemetry.get('composite_score', telemetry.get('confidence', 0.50)))
            ambiguous_reason = f'Ambiguous VLM assessment: contact={contact}, damage={damage}, displacement={displacement}, benign={benign_alt}'
            logger.info(f'Candidate @ F{candidate_frame} quarantined: {ambiguous_reason}')
            result = {
                'verdict': 'CANDIDATE_UNCONFIRMED',
                'is_collision': False,
                'collision_type': str(telemetry.get('event_type', 'unconfirmed')),
                'confidence': round(cand_conf, 2),
                'visual_cues': [],
                'rationale': f'Quarantined: {ambiguous_reason}',
                'source': f'{vlm_source}_quarantined',
                'suppressed': True,
                'suppression_reason': ambiguous_reason,
                'vlm_evaluation': vlm_data,
                'bypass_telemetry': bypass_tel,
            }
        self._cached_results[cache_key] = result
        return result

    def _evaluate_deterministic_high_confidence_bypass(
        self,
        telemetry: Dict[str, Any],
        candidate_frame: int,
        video_id: str,
    ) -> Tuple[bool, str, Dict[str, Any], str]:
        event_type = str(telemetry.get('event_type', 'collision'))
        accel = float(telemetry.get('accel_ms2', telemetry.get('a_long', telemetry.get('acceleration_mag', 0.0))))
        abs_accel = abs(accel)
        dist = float(telemetry.get('distance_m', telemetry.get('ground_distance_m', telemetry.get('dca_m', 99.0))))
        iou = float(telemetry.get('bbox_iou', telemetry.get('iou', 0.0)))
        max_spd = float(telemetry.get('max_recent_speed_kmh', telemetry.get('speed_kmh', telemetry.get('metric_speed_kmh', 0.0))))
        curr_spd = float(telemetry.get('speed_kmh', telemetry.get('metric_speed_kmh', 0.0)))
        speed_drop = float(telemetry.get('delta_v_drop', max(0.0, max_spd - curr_spd)))
        post_rest = bool(telemetry.get('post_event_rest', False))
        is_crossing = bool(telemetry.get('is_crossing_trajectory', False))
        s_dissip = float(telemetry.get('S_dissip', 0.0))
        yaw_delta = max(float(telemetry.get('yaw_delta_deg', 0.0)), float(telemetry.get('theta_deg', 0.0)))
        d_edge_m = float(telemetry.get('d_edge_m', 99.0))
        shoulder_admitted = bool(telemetry.get('shoulder_admitted', False))
        v_dot_n = float(telemetry.get('v_dot_n', 0.0))

        # Real measured stationary duration (NO synthetic imputation)
        tau_rest_real = float(telemetry.get('stationary_duration_sec', telemetry.get('rest_duration_s', telemetry.get('tau_rest', 0.0))))
        norm_px = float(telemetry.get('norm_pixel_dist', 0.0 if ('bbox_iou' in telemetry or 'iou' in telemetry) and 'norm_pixel_dist' not in telemetry and 'distance_m' not in telemetry else 99.0))
        is_closure_corridor = bool(telemetry.get('is_kinematic_closure_corridor', False))
        is_calibrated = bool(telemetry.get('homography_calibrated', telemetry.get('is_calibrated', 'distance_m' in telemetry or 'ground_distance_m' in telemetry)))
        min_track_age = int(telemetry.get('track_age', 10))

        bypass_tel = {
            'candidate_frame': candidate_frame,
            'event_type': event_type,
            'contact_evidence': {
                'bbox_iou': round(iou, 3),
                'distance_m': round(dist, 3),
                'norm_pixel_dist': round(norm_px, 3),
                'is_crossing': is_crossing,
                'is_kinematic_closure_corridor': is_closure_corridor,
                'S_dissip': round(s_dissip, 3),
                'is_calibrated': is_calibrated,
            },
            'impulse': {
                'accel_ms2': round(accel, 2),
                'abs_accel_ms2': round(abs_accel, 2),
            },
            'velocities': {
                'pre_speed_kmh': round(max_spd, 2),
                'curr_speed_kmh': round(curr_spd, 2),
                'speed_drop_kmh': round(speed_drop, 2),
                'delta_v_drop': round(speed_drop, 2),
            },
            'rest_and_arrest': {
                'post_event_rest': post_rest,
                'rest_duration_s': round(tau_rest_real, 2),
                'tau_rest': round(tau_rest_real, 2),
                'tau_rest_real': round(tau_rest_real, 2),
            },
            'barrier_geometry': {
                'd_boundary_m': round(d_edge_m, 3),
                'v_dot_n': round(v_dot_n, 3),
                'shoulder_admitted': shoulder_admitted,
            },
            'track_maturity': {
                'min_track_age': min_track_age,
            },
            'bypass_passed': False,
            'bypass_reason': '',
        }

        # 1. Confirmed Wrong-Way Vehicle Hazard
        if event_type in ('wrong_way_hazard', 'wrong_way_vehicle'):
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = 'confirmed_wrong_way_hazard'
            return True, 'wrong_way_hazard', bypass_tel, f'Confirmed wrong-way hazard contrary to dominant roadway flow @ F{candidate_frame}'

        # 2. Multi-vehicle physical collisions
        is_single = event_type in ('single_vehicle_anomaly', 'single', 'barrier_strike', 'single_vehicle_collision') or telemetry.get('is_single_vehicle', False)
        if not is_single:
            # Canonical Multi-Vehicle Deterministic Bypass Gate
            has_spatial_contact = bool(
                (is_calibrated and dist <= 2.0 and iou >= 0.15)
                or (not is_calibrated and norm_px <= 0.50 and iou >= 0.30)
            )
            has_signed_decel = bool(accel <= -12.0)
            has_finite_dissipation = bool(speed_drop >= 15.0)
            has_unimputed_rest = bool(tau_rest_real >= 0.25)
            has_track_maturity = bool(min_track_age >= 5)

            is_canonical_multi_vehicle_bypass = bool(
                has_spatial_contact
                and has_signed_decel
                and has_finite_dissipation
                and has_unimputed_rest
                and has_track_maturity
            )

            if is_canonical_multi_vehicle_bypass:
                coll_type = 't_bone_collision' if is_crossing else 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = (
                    f'canonical_multi_vehicle_bypass (dist={dist:.2f}, norm_px={norm_px:.2f}, '
                    f'iou={iou:.2f}, a_long={accel:+.1f}, dv_drop={speed_drop:.1f}, '
                    f'tau_rest={tau_rest_real:.2f}, age={min_track_age})'
                )
                return (
                    True,
                    coll_type,
                    bypass_tel,
                    f'Canonical multi-vehicle impact bypass: contact=(dist={dist:.2f}m, norm_px={norm_px:.2f}, iou={iou:.2f}), '
                    f'a_long={accel:+.1f} <= -12 m/s^2, dv_drop={speed_drop:.1f} >= 15 km/h, tau_rest={tau_rest_real:.2f}s >= 0.25s, age={min_track_age}'
                )

            # Adverse weather multi-vehicle bypass (remediated with signed decel and track age)
            weather = str(telemetry.get('weather', 'normal')).lower()
            is_adverse = any(w in weather for w in ('rain', 'snow', 'fog', 'wet'))
            if (
                is_adverse
                and min_track_age >= 5
                and accel <= -12.0
                and (speed_drop >= 15.0 or s_dissip >= 0.70)
                and tau_rest_real >= 0.25
                and (
                    (is_calibrated and dist <= 2.5 and iou >= 0.15)
                    or (not is_calibrated and norm_px <= 0.60 and iou >= 0.20)
                )
            ):
                coll_type = 't_bone_collision' if (is_crossing or s_dissip >= 0.50) else 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'adverse_weather_high_impulse (weather={weather}, dist={dist:.2f}, norm_px={norm_px:.2f}, accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Adverse weather collision: weather={weather}, impulse={accel:+.1f} m/s^2, dv_drop={speed_drop:.1f}'

            bypass_tel['bypass_reason'] = 'multi_vehicle_insufficient_physical_evidence'
            return False, 'none', bypass_tel, 'Multi-vehicle candidate lacks safety-critical impact evidence'

        # 3. Single-vehicle incidents (barrier strikes, spin-outs, rollovers)
        v_entry = float(telemetry.get('v_entry_kmh', max_spd))
        anomaly_type = str(telemetry.get('anomaly_type', '')).lower()
        if anomaly_type == 'yaw' and not any(k in anomaly_type for k in ('composite', 'barrier', 'rollover', 'decel_stop')):
            bypass_tel['bypass_reason'] = 'benign_turning_maneuver_yaw_only'
            return False, 'none', bypass_tel, 'Rejected by bypass: turning maneuver with yaw-only anomaly'

        s_single = float(telemetry.get('composite_score', telemetry.get('s_single', 0.0)))
        has_severe_crash_impulse = (accel <= -15.0)
        has_violent_speed_loss = (speed_drop >= 20.0 and v_entry >= 20.0)
        is_near_boundary = (d_edge_m <= 0.8 or shoulder_admitted)
        is_boundary_aligned = (v_dot_n >= 0.35)
        has_single_rest = (tau_rest_real >= 1.0)
        has_single_maturity = (min_track_age >= 5)

        # Remediated severe_barrier_impact_dissipation
        if (
            s_single >= 0.48
            and is_near_boundary
            and is_boundary_aligned
            and has_severe_crash_impulse
            and has_violent_speed_loss
            and has_single_rest
            and has_single_maturity
        ):
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'severe_barrier_impact_dissipation (s_single={s_single:.2f}, d_edge={d_edge_m:.2f}, v_dot_n={v_dot_n:.2f}, accel={accel:.1f}, drop={speed_drop:.1f}, tau_rest={tau_rest_real:.2f})'
            return True, 'barrier_strike', bypass_tel, f'High-energy barrier impact / run-off-road: S_single={s_single:.2f}, d_edge={d_edge_m:.2f}m, impulse={accel:+.1f} m/s^2, speed_drop={speed_drop:.1f} km/h, tau_rest={tau_rest_real:.2f}s'

        # Remediated severe_spin_out
        is_spinout_type = any(k in anomaly_type for k in ('composite', 'rollover'))
        is_plausible_spinout_speed = (35.0 <= v_entry <= 110.0 and 20.0 <= speed_drop <= 75.0)
        if (
            is_spinout_type
            and s_single >= 0.65
            and yaw_delta >= 50.0
            and is_plausible_spinout_speed
            and has_severe_crash_impulse
            and curr_spd <= 18.0
            and has_single_rest
            and has_single_maturity
        ):
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'severe_spin_out (yaw={yaw_delta:.1f}, drop={speed_drop:.1f}, spd={curr_spd:.1f}, tau_rest={tau_rest_real:.2f})'
            return True, 'spin_out', bypass_tel, f'Severe spin-out with trajectory disruption: yaw_delta={yaw_delta:.1f}deg, speed_drop={speed_drop:.1f} km/h, tau_rest={tau_rest_real:.2f}s'

        # Remediated post_event_rest
        is_arrested_spinout = (
            curr_spd <= 6.0
            and s_single >= 0.60
            and yaw_delta >= 90.0
            and v_entry >= 35.0
            and speed_drop >= 25.0
            and (d_edge_m <= 8.0 or shoulder_admitted)
            and has_single_rest
            and has_single_maturity
        )
        is_impact_rest = (
            curr_spd <= 5.0
            and has_severe_crash_impulse
            and speed_drop >= 20.0
            and v_entry >= 20.0
            and (d_edge_m <= 2.0 or shoulder_admitted)
            and s_single >= 0.65
            and has_single_rest
            and has_single_maturity
        )
        if is_arrested_spinout or is_impact_rest:
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'post_event_rest (spd={curr_spd:.1f}, yaw={yaw_delta:.1f}, drop={speed_drop:.1f}, tau_rest={tau_rest_real:.2f})'
            return True, 'spin_out', bypass_tel, f'Post-event vehicle arrest: curr_spd={curr_spd:.1f} km/h, yaw_delta={yaw_delta:.1f}deg, tau_rest={tau_rest_real:.2f}s'

        bypass_tel['bypass_reason'] = 'single_vehicle_insufficient_impact_evidence'
        return False, 'none', bypass_tel, 'Single-vehicle candidate lacks high-confidence physical impact evidence'


    def _evaluate_and_dispatch_fallback(
        self,
        telemetry: Dict[str, Any],
        candidate_frame: int,
        video_id: str,
    ) -> Dict[str, Any]:
        is_bypass, coll_type, bypass_tel, rationale = self._evaluate_deterministic_high_confidence_bypass(
            telemetry=telemetry,
            candidate_frame=candidate_frame,
            video_id=video_id,
        )
        cache_key = self._make_cache_key(video_id, candidate_frame, telemetry)
        if is_bypass:
            logger.info(f'DETERMINISTIC BYPASS ACTIVATED @ F{candidate_frame}: {rationale}')
            result = {
                'verdict': 'CONFIRMED_INCIDENT',
                'is_collision': True,
                'collision_type': coll_type,
                'confidence': 0.95,
                'visual_cues': [
                    'physical contact / high bounding box overlap',
                    'severe acceleration impulse beyond friction limits',
                    'post-event kinetic dissipation / trajectory arrest',
                ],
                'rationale': f'Deterministic high-confidence bypass confirmed incident at F{candidate_frame}: {rationale}',
                'source': 'deterministic_high_confidence_bypass',
                'suppressed': False,
                'bypass_telemetry': bypass_tel,
            }
        else:
            cand_conf = float(telemetry.get('composite_score', telemetry.get('confidence', 0.50)))
            suppression_reason = bypass_tel.get('bypass_reason', 'unverified_offline_fallback')
            logger.info(f'VLM OFFLINE FALLBACK SUPPRESSED @ F{candidate_frame} -> CANDIDATE_UNCONFIRMED ({suppression_reason})')
            result = {
                'verdict': 'CANDIDATE_UNCONFIRMED',
                'is_collision': False,
                'collision_type': str(telemetry.get('event_type', 'unconfirmed')),
                'confidence': round(cand_conf, 2),
                'visual_cues': [],
                'rationale': f'VLM unavailable/offline; candidate does not qualify for deterministic_high_confidence_bypass ({suppression_reason}).',
                'source': 'offline_unverified_fallback',
                'suppressed': True,
                'suppression_reason': suppression_reason,
                'bypass_telemetry': bypass_tel,
            }
        self._cached_results[cache_key] = result
        return result

    def _generate_fallback_verdict(
        self,
        telemetry: Dict[str, Any],
        candidate_frame: int,
        video_id: str,
    ) -> Dict[str, Any]:
        return self._evaluate_and_dispatch_fallback(telemetry, candidate_frame, video_id)
