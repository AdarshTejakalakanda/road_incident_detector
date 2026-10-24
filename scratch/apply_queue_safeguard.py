with open("road_surveillance/events/candidates/collision.py", "r", encoding="utf-8") as f:
    code = f.read()

old_block = """            # Proposal A: Terminal Physical-Contact Condition
            has_bbox_contact = bool(iou >= 0.15 or (iou >= 0.10 and iou_x >= 0.50))
            has_calibrated_ground_contact = bool(self.is_camera_calibrated and dist <= 2.0)
            has_uncalibrated_pixel_proximity = bool(not self.is_camera_calibrated and (norm_pixel_dist <= 0.50 or (norm_pixel_dist <= 0.85 and iou > 0.0)))

            has_terminal_physical_contact = bool(
                has_bbox_contact or has_calibrated_ground_contact or has_uncalibrated_pixel_proximity
            )

            has_contact_distance_sanity = bool(
                (dist <= 3.5) if self.is_camera_calibrated else (norm_pixel_dist <= 1.0 or dist <= 4.0)
            )"""

new_block = """            # Proposal A: Terminal Physical-Contact Condition
            b1_bottom = b1[1] + b1[3]
            b2_bottom = b2[1] + b2[3]
            dy_bottom = abs(b1_bottom - b2_bottom)
            min_h = min(b1[3], b2[3])
            norm_bottom_dy = dy_bottom / max(1.0, min_h)

            # In same-direction traffic, perspective vertical overlap between a car and a car far ahead in the queue is NOT a crash.
            # If the car behind is separated down the roadway (dist > 4.5m or norm_bottom_dy > 0.40), it is normal queue following.
            is_queue_following = is_parallel_heading and (dist > 4.5 or norm_bottom_dy > 0.40) and not (iou >= 0.35 and (severe_decel or s_v_drop >= 0.50))

            has_bbox_contact = bool((iou >= 0.15 or (iou >= 0.10 and iou_x >= 0.50)) and not is_queue_following)
            has_calibrated_ground_contact = bool(self.is_camera_calibrated and dist <= 2.0)
            has_uncalibrated_pixel_proximity = bool(not self.is_camera_calibrated and (norm_pixel_dist <= 0.50 or (norm_pixel_dist <= 0.85 and iou > 0.0)) and not is_queue_following)

            has_terminal_physical_contact = bool(
                has_bbox_contact or has_calibrated_ground_contact or has_uncalibrated_pixel_proximity
            )

            has_contact_distance_sanity = bool(
                (dist <= 3.5) if self.is_camera_calibrated else (norm_pixel_dist <= 1.0 and dist <= 4.5)
            )"""

assert old_block in code, "old_block not found in collision.py"
code = code.replace(old_block, new_block, 1)

with open("road_surveillance/events/candidates/collision.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully updated collision.py with queue-following safeguard!")
