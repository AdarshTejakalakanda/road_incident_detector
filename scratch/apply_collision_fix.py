with open("road_surveillance/events/candidates/collision.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Vehicle class check
old_cls = """            # Strict vehicle class enforcement
            if tr1.class_label.lower() not in self.VEHICLE_LABELS or tr2.class_label.lower() not in self.VEHICLE_LABELS:
                continue"""

new_cls = """            # Strict vehicle class enforcement
            lbl1 = getattr(tr1, "get_effective_label", lambda: tr1.class_label)().lower()
            lbl2 = getattr(tr2, "get_effective_label", lambda: tr2.class_label)().lower()
            if (lbl1 not in self.VEHICLE_LABELS and tr1.class_label.lower() not in self.VEHICLE_LABELS) or (lbl2 not in self.VEHICLE_LABELS and tr2.class_label.lower() not in self.VEHICLE_LABELS):
                continue"""

assert old_cls in code, "old_cls not found"
code = code.replace(old_cls, new_cls, 1)

# 2. Max collision distance sanity check (lines 324-330)
old_dist_sanity = """            # Max collision distance sanity check:
            if not is_kinematic_closure_corridor:
                if dist > selected_max_dist_m and iou < 0.10:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

new_dist_sanity = """            # Max collision distance sanity check:
            if not is_kinematic_closure_corridor:
                has_prox_sanity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not has_prox_sanity and iou < 0.10:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

assert old_dist_sanity in code, "old_dist_sanity not found"
code = code.replace(old_dist_sanity, new_dist_sanity, 1)

# 3. Duplicate check and deceleration calculation
old_dup_decel = """            # Safeguard 2: Duplicate tracks on same vehicle
            # Two tracks on the same physical vehicle travel in the same direction, have high overlap,
            # and their distance is not closing from far away.
            is_duplicate = (iou >= 0.25 or (dist < 1.2 and norm_pixel_dist <= 0.35)) and heading_aligned and not signal_decreasing_dist
            if is_duplicate:
                self.persistence_counts[pair_key] = 0
                continue

            # Signal 3: Deceleration anomaly
            decel_1 = k1.acceleration_mag < self.deceleration_threshold_ms2
            decel_2 = k2.acceleration_mag < self.deceleration_threshold_ms2
            signal_sudden_decel = bool(decel_1 or decel_2)
            severe_decel = bool(k1.acceleration_mag < -6.0 or k2.acceleration_mag < -6.0)"""

new_dup_decel = """            # Signal 3: Deceleration anomaly
            decel_1 = k1.acceleration_mag < self.deceleration_threshold_ms2
            decel_2 = k2.acceleration_mag < self.deceleration_threshold_ms2
            signal_sudden_decel = bool(decel_1 or decel_2)
            severe_decel = bool(k1.acceleration_mag < -6.0 or k2.acceleration_mag < -6.0)

            # Safeguard 2: Duplicate tracks on same vehicle
            # Two tracks on the same physical vehicle travel in the same direction, have high overlap,
            # and their distance is not closing from far away.
            is_duplicate = (iou >= 0.25 or (dist < 1.2 and norm_pixel_dist <= 0.35)) and heading_aligned and not signal_decreasing_dist
            if is_duplicate:
                # Genuine physical collision exhibits severe decel or kinetic speed collapse; do not suppress as duplicate box
                if not (severe_decel or s_v_drop >= 0.50 or signal_sudden_decel):
                    self.persistence_counts[pair_key] = 0
                    continue"""

assert old_dup_decel in code, "old_dup_decel not found"
code = code.replace(old_dup_decel, new_dup_decel, 1)

# 4. Multi-vehicle collision ground distance gate & physical approach
old_ground_gate = """            # Multi-vehicle collision ground distance gate (d_ground <= selected_max_dist_m or physical impact or kinematic closure corridor)
            if not is_kinematic_closure_corridor:
                if dist > selected_max_dist_m and not (iou >= 0.20 and severe_decel and (signal_decreasing_dist or signal_convergence)):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue

            # Multi-signal triggering rule
            # In aligned/parallel flow, require genuine physical approach rather than 2D perspective box overlap
            if heading_aligned:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach)
            else:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach or (iou >= 0.20))"""

new_ground_gate = """            # Multi-vehicle collision ground distance gate (d_ground <= selected_max_dist_m or physical impact or kinematic closure corridor)
            if not is_kinematic_closure_corridor:
                is_physical_impact = bool(iou >= 0.20 and (severe_decel or s_v_drop >= 0.50))
                has_valid_proximity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not (has_valid_proximity or is_physical_impact):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue

            # Multi-signal triggering rule
            # In aligned/parallel flow, require genuine physical approach rather than 2D perspective box overlap
            if heading_aligned:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach or (iou >= 0.20 and (severe_decel or s_v_drop >= 0.50)))
            else:
                has_physical_approach = bool(signal_decreasing_dist or signal_convergence or has_angular_approach or (iou >= 0.20))"""

assert old_ground_gate in code, "old_ground_gate not found"
code = code.replace(old_ground_gate, new_ground_gate, 1)

with open("road_surveillance/events/candidates/collision.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully applied updates to collision.py!")
