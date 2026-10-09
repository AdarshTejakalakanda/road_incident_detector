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
        speed_drop = max(0.0, max_spd - curr_spd)
        post_rest = bool(telemetry.get('post_event_rest', False))
        is_crossing = bool(telemetry.get('is_crossing_trajectory', False))
        s_dissip = float(telemetry.get('S_dissip', 0.0))
        yaw_delta = max(float(telemetry.get('yaw_delta_deg', 0.0)), float(telemetry.get('theta_deg', 0.0)))
        d_edge_m = float(telemetry.get('d_edge_m', 99.0))
        shoulder_admitted = bool(telemetry.get('shoulder_admitted', False))
        v_dot_n = float(telemetry.get('v_dot_n', 0.0))
        tau_rest = float(telemetry.get('tau_rest', telemetry.get('stationary_duration_sec', telemetry.get('rest_duration_s', 0.0))))
        if tau_rest < 0.25 and (curr_spd <= 5.0 or post_rest):
            tau_rest = 0.25
        trajectory_arrest = bool(curr_spd <= 6.0 or post_rest or tau_rest >= 0.25)
        norm_px = float(telemetry.get('norm_pixel_dist', 99.0))
        is_closure_corridor = bool(telemetry.get('is_kinematic_closure_corridor', False))
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
            },
            'impulse': {
                'accel_ms2': round(accel, 2),
                'abs_accel_ms2': round(abs_accel, 2),
            },
            'velocities': {
                'pre_speed_kmh': round(max_spd, 2),
                'curr_speed_kmh': round(curr_spd, 2),
                'speed_drop_kmh': round(speed_drop, 2),
            },
            'rest_and_arrest': {
                'post_event_rest': post_rest,
                'rest_duration_s': round(tau_rest, 2),
                'tau_rest': round(tau_rest, 2),
                'trajectory_arrest': trajectory_arrest,
            },
            'barrier_geometry': {
                'd_boundary_m': round(d_edge_m, 3),
                'v_dot_n': round(v_dot_n, 3),
                'shoulder_admitted': shoulder_admitted,
            },
            'bypass_passed': False,
            'bypass_reason': '',
        }

        # Stage T Section 7 Primary Deterministic Bypass Rule
        if (iou >= 0.25) and (accel <= -12.0) and (tau_rest >= 0.25):
            coll_type = 't_bone_collision' if is_crossing else 'rear_end_collision'
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'stage_t_deterministic_bypass (iou={iou:.2f}, a_long={accel:+.1f}, tau_rest={tau_rest:.2f})'
            return True, coll_type, bypass_tel, f'Deterministic high-confidence bypass: IoU={iou:.2f} >= 0.25, a_long={accel:+.1f} <= -12 m/s^2, tau_rest={tau_rest:.2f}s >= 0.25s'

        # 1. Confirmed Wrong-Way Vehicle Hazard
        if event_type in ('wrong_way_hazard', 'wrong_way_vehicle'):
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = 'confirmed_wrong_way_hazard'
            return True, 'wrong_way_hazard', bypass_tel, f'Confirmed wrong-way hazard contrary to dominant roadway flow @ F{candidate_frame}'

        # 2. Multi-vehicle physical collisions
        is_single = event_type in ('single_vehicle_anomaly', 'single', 'barrier_strike', 'single_vehicle_collision') or telemetry.get('is_single_vehicle', False)
        if not is_single:
            curr_spd_ms = curr_spd / 3.6
            if abs_accel >= 0.5 and curr_spd_ms > 0.0 and iou < 0.10:
                if (curr_spd_ms / abs_accel) < 0.10:
                    bypass_tel['bypass_reason'] = 'kalman_jitter_implausible_time_to_rest'
                    return False, 'none', bypass_tel, 'Rejected by bypass: Kalman jitter implausible time to rest'
            if iou >= 0.30 and (abs_accel >= 15.0 or accel <= -12.0):
                coll_type = 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'multi_vehicle_high_iou_impulse (iou={iou:.2f}, accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Severe multi-vehicle physical impact: iou={iou:.2f}, impulse={accel:+.1f} m/s^2'
            if dist <= 1.8 and iou >= 0.10 and (abs_accel >= 15.0 or accel <= -12.0):
                coll_type = 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'multi_vehicle_contact_impulse (dist={dist:.2f}, iou={iou:.2f}, accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Close-contact multi-vehicle collision: dist={dist:.2f}m, iou={iou:.2f}, impulse={accel:+.1f} m/s^2'
            if (is_crossing or s_dissip >= 0.55) and dist <= 3.2 and (abs_accel >= 15.0 or accel <= -12.0) and (iou >= 0.10 or s_dissip >= 0.70):
                coll_type = 't_bone_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'crossing_tbone_high_dissipation (dist={dist:.2f}, S_dissip={s_dissip:.2f}, accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Crossing T-bone collision: dist={dist:.2f}m, S_dissip={s_dissip:.2f}, impulse={accel:+.1f} m/s^2'
            weather = str(telemetry.get('weather', 'normal')).lower()
            is_adverse = any(w in weather for w in ('rain', 'snow', 'fog', 'wet'))
            if is_adverse and (dist <= 3.5 or norm_px <= 1.5) and (iou >= 0.25 or abs_accel >= 20.0 or (s_dissip >= 0.70 and abs_accel >= 15.0)):
                coll_type = 't_bone_collision' if (is_crossing or s_dissip >= 0.50) else 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'adverse_weather_high_impulse (weather={weather}, dist={dist:.2f}, norm_px={norm_px:.2f}, accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Adverse weather collision: weather={weather}, impulse={accel:+.1f} m/s^2'
            if is_closure_corridor and (abs_accel >= 15.0 or accel <= -12.0):
                coll_type = 'rear_end_collision'
                bypass_tel['bypass_passed'] = True
                bypass_tel['bypass_reason'] = f'closure_corridor_impulse (accel={accel:.1f})'
                return True, coll_type, bypass_tel, f'Persistent parallel closure impact: impulse={accel:+.1f} m/s^2'
            bypass_tel['bypass_reason'] = 'multi_vehicle_insufficient_physical_evidence'
            return False, 'none', bypass_tel, 'Multi-vehicle candidate lacks safety-critical impact evidence'

        # 3. Single-vehicle incidents (barrier strikes, spin-outs, rollovers)
        anomaly_type = str(telemetry.get('anomaly_type', '')).lower()
        if anomaly_type == 'yaw' and not any(k in anomaly_type for k in ('composite', 'barrier', 'rollover', 'decel_stop')):
            bypass_tel['bypass_reason'] = 'benign_turning_maneuver_yaw_only'
            return False, 'none', bypass_tel, 'Rejected by bypass: turning maneuver with yaw-only anomaly'
        curr_spd_ms = curr_spd / 3.6
        if abs_accel >= 10.0 and curr_spd_ms > 0.0:
            time_to_rest_s = curr_spd_ms / abs_accel
            if time_to_rest_s < 0.15 and yaw_delta < 45.0 and speed_drop < 20.0:
                bypass_tel['bypass_reason'] = 'kalman_jitter_implausible_time_to_rest'
                return False, 'none', bypass_tel, 'Rejected by bypass: Kalman jitter implausible time to rest'
        s_single = float(telemetry.get('composite_score', telemetry.get('s_single', 0.0)))
        has_severe_crash_impulse = (abs_accel >= 25.0 or accel <= -25.0)
        has_violent_speed_loss = (speed_drop >= 20.0 and max_spd >= 30.0)
        is_near_boundary = (d_edge_m <= 0.8 or shoulder_admitted)
        is_boundary_aligned = (v_dot_n >= 0.35)
        if s_single >= 0.48 and is_near_boundary and is_boundary_aligned and has_severe_crash_impulse and has_violent_speed_loss:
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'severe_barrier_impact_dissipation (s_single={s_single:.2f}, d_edge={d_edge_m:.2f}, v_dot_n={v_dot_n:.2f}, accel={accel:.1f}, drop={speed_drop:.1f})'
            return True, 'barrier_strike', bypass_tel, f'High-energy barrier impact / run-off-road: S_single={s_single:.2f}, d_edge={d_edge_m:.2f}m, impulse={accel:+.1f} m/s^2, speed_drop={speed_drop:.1f} km/h'
        is_spinout_type = any(k in anomaly_type for k in ('composite', 'rollover'))
        is_plausible_spinout_speed = (35.0 <= max_spd <= 110.0 and 20.0 <= speed_drop <= 75.0)
        if is_spinout_type and s_single >= 0.65 and yaw_delta >= 50.0 and is_plausible_spinout_speed and abs_accel >= 15.0 and curr_spd <= 18.0:
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'severe_spin_out (yaw={yaw_delta:.1f}, drop={speed_drop:.1f}, spd={curr_spd:.1f})'
            return True, 'spin_out', bypass_tel, f'Severe spin-out with trajectory disruption: yaw_delta={yaw_delta:.1f}deg, speed_drop={speed_drop:.1f} km/h'
        is_arrested_spinout = (
            curr_spd <= 6.0
            and s_single >= 0.60
            and yaw_delta >= 90.0
            and max_spd >= 35.0
            and speed_drop >= 25.0
            and (d_edge_m <= 8.0 or shoulder_admitted)
        )
        is_impact_rest = (curr_spd <= 5.0 and abs_accel >= 15.0 and speed_drop >= 20.0 and (d_edge_m <= 2.0 or shoulder_admitted) and s_single >= 0.65)
        if is_arrested_spinout or is_impact_rest:
            bypass_tel['bypass_passed'] = True
            bypass_tel['bypass_reason'] = f'post_event_rest (spd={curr_spd:.1f}, yaw={yaw_delta:.1f}, drop={speed_drop:.1f})'
            return True, 'spin_out', bypass_tel, f'Post-event vehicle arrest: curr_spd={curr_spd:.1f} km/h, yaw_delta={yaw_delta:.1f}deg'
        bypass_tel['bypass_reason'] = 'single_vehicle_insufficient_impact_evidence'
        return False, 'none', bypass_tel, 'Single-vehicle candidate lacks high-confidence physical impact evidence'

    