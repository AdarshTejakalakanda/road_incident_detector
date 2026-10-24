with open("road_surveillance/events/candidates/collision.py", "r", encoding="utf-8") as f:
    code = f.read()

old_338 = """            # Max collision distance sanity check:
            if not is_kinematic_closure_corridor:
                has_prox_sanity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not has_prox_sanity and iou < 0.10:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

new_338 = """            # Max collision distance sanity check:
            if not is_kinematic_closure_corridor:
                has_prox_sanity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and dist <= 6.0 and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not has_prox_sanity and iou < 0.10:
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

assert old_338 in code, "old_338 not found"
code = code.replace(old_338, new_338, 1)

old_558 = """            # Multi-vehicle collision ground distance gate (d_ground <= selected_max_dist_m or physical impact or kinematic closure corridor)
            if not is_kinematic_closure_corridor:
                is_physical_impact = bool(iou >= 0.20 and (severe_decel or s_v_drop >= 0.50))
                has_valid_proximity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not (has_valid_proximity or is_physical_impact):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

new_558 = """            # Multi-vehicle collision ground distance gate (d_ground <= selected_max_dist_m or physical impact or kinematic closure corridor)
            if not is_kinematic_closure_corridor:
                is_physical_impact = bool(iou >= 0.20 and (severe_decel or s_v_drop >= 0.50) and dist <= 6.0)
                has_valid_proximity = bool(dist <= selected_max_dist_m) or (not self.is_camera_calibrated and dist <= 6.0 and (norm_pixel_dist <= 0.85 or iou >= 0.15))
                if not (has_valid_proximity or is_physical_impact):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "collision_distance_exceeded", dist)
                    continue"""

assert old_558 in code, "old_558 not found"
code = code.replace(old_558, new_558, 1)

with open("road_surveillance/events/candidates/collision.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully added dist <= 6.0 sanity bounds in collision.py!")
