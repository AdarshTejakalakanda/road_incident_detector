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
        norm_px = float(telemetry.get('norm_pixel_dist', 99.0))
        is_closure_corridor = bool(telemetry.get('is_kinematic_closure_corridor', False))
        is_calibrated = bool(telemetry.get('homography_calibrated', telemetry.get('is_calibrated', False)))
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
                or (not is_calibrated and (norm_px <= 0.50 or dist <= 2.0 or norm_px > 90.0) and iou >= 0.30)
            )
            has_signed_decel = bool(accel <= -12.0)
            has_finite_dissipation = bool(speed_drop >= 15.0 or s_dissip >= 0.70 or abs_accel >= 18.0)
            has_unimputed_rest = bool(tau_rest_real >= 0.25 or (abs_accel >= 18.0 and dist <= 1.0))
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
