with open("modules/vlm_arbitrator.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update speed_drop and is_calibrated
old_calib = """        speed_drop = float(telemetry.get('delta_v_drop', max(0.0, max_spd - curr_spd)))
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
        is_calibrated = bool(telemetry.get('homography_calibrated', telemetry.get('is_calibrated', 'distance_m' in telemetry or 'ground_distance_m' in telemetry)))"""

new_calib = """        speed_drop = float(telemetry.get('delta_v_drop', telemetry.get('speed_drop_kmh', max(0.0, max_spd - curr_spd))))
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
        is_calibrated = bool(telemetry.get('homography_calibrated', telemetry.get('is_calibrated', False)))"""

assert old_calib in content, "old_calib not found"
content = content.replace(old_calib, new_calib, 1)

# 2. Update multi-vehicle spatial contact
old_contact = """            # Canonical Multi-Vehicle Deterministic Bypass Gate
            has_spatial_contact = bool(
                (is_calibrated and dist <= 2.0 and iou >= 0.15)
                or (not is_calibrated and norm_px <= 0.50 and iou >= 0.30)
            )
            has_signed_decel = bool(accel <= -12.0)
            has_finite_dissipation = bool(speed_drop >= 15.0)
            has_unimputed_rest = bool(tau_rest_real >= 0.25)
            has_track_maturity = bool(min_track_age >= 5)"""

new_contact = """            # Canonical Multi-Vehicle Deterministic Bypass Gate
            has_spatial_contact = bool(
                (is_calibrated and dist <= 2.0 and iou >= 0.15)
                or (not is_calibrated and (norm_px <= 0.60 or iou >= 0.25) and iou >= 0.15)
                or (iou >= 0.35)
            )
            has_signed_decel = bool(accel <= -12.0)
            has_finite_dissipation = bool(speed_drop >= 15.0 or s_dissip >= 0.50)
            has_unimputed_rest = bool(tau_rest_real >= 0.25)
            has_track_maturity = bool(min_track_age >= 5)"""

assert old_contact in content, "old_contact not found"
content = content.replace(old_contact, new_contact, 1)

with open("modules/vlm_arbitrator.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated modules/vlm_arbitrator.py!")
