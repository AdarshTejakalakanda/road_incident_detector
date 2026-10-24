with open("run_pipeline.py", "r", encoding="utf-8") as f:
    code = f.read()

old_tel = """        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -4.0)))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        iou_val = float(cand_signals.get("bbox_iou", 0.10))

        telemetry = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "event_type": cand_type,
            "involved_tracks": cand["track_ids"],
            "composite_score": float(cand_signals.get("composite_score", 0.65)),
            "accel_ms2": acc_val,
            "yaw_delta_deg": float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 30.0))),
            "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
            "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
            "distance_m": dist_val,
            "bbox_iou": iou_val,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": float(cand_signals.get("max_recent_speed_kmh", 30.0)),
            "speed_kmh": float(cand_signals.get("metric_speed_kmh", 5.0)),
            "track_age": 10,
            "is_single_vehicle": is_single,
            "post_event_rest": cand_signals.get("post_event_rest", True),
            "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
            "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
            "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
        }"""

new_tel = """        acc_1 = float(cand_signals.get("accel_1_ms2", 0.0))
        acc_2 = float(cand_signals.get("accel_2_ms2", 0.0))
        min_acc = min(acc_1, acc_2) if (acc_1 != 0.0 or acc_2 != 0.0) else -4.0
        acc_val = float(cand_signals.get("metric_accel_ms2", min_acc))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        iou_val = float(cand_signals.get("bbox_iou", 0.10))
        norm_px = float(cand_signals.get("norm_pixel_dist", 0.35))
        is_calib = bool(cand_signals.get("is_calibrated", camera_cfg.is_calibrated))
        tau_rest = float(cand_signals.get("tau_rest", cand_signals.get("stationary_duration_sec", 0.0)))
        spd_1 = float(cand_signals.get("speed_1_kmh", 0.0))
        spd_2 = float(cand_signals.get("speed_2_kmh", 0.0))
        max_spd = max(spd_1, spd_2, float(cand_signals.get("max_recent_speed_kmh", 30.0)))
        curr_spd = min(spd_1, spd_2) if (spd_1 > 0 or spd_2 > 0) else float(cand_signals.get("metric_speed_kmh", 0.0))
        speed_drop = max(0.0, max_spd - curr_spd)

        telemetry = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "event_type": cand_type,
            "involved_tracks": cand["track_ids"],
            "composite_score": float(cand_signals.get("composite_score", 0.85)),
            "accel_ms2": acc_val,
            "yaw_delta_deg": float(cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 30.0))),
            "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
            "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
            "distance_m": dist_val,
            "norm_pixel_dist": norm_px,
            "bbox_iou": iou_val,
            "is_calibrated": is_calib,
            "homography_calibrated": is_calib,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": max_spd,
            "speed_kmh": curr_spd,
            "speed_drop_kmh": speed_drop,
            "tau_rest": tau_rest,
            "stationary_duration_sec": tau_rest,
            "track_age": 15,
            "is_single_vehicle": is_single,
            "post_event_rest": cand_signals.get("post_event_rest", True),
            "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
            "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
            "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
        }"""

assert old_tel in code, "old_tel not found in run_pipeline.py"
code = code.replace(old_tel, new_tel, 1)

with open("run_pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully updated telemetry packing in run_pipeline.py!")
