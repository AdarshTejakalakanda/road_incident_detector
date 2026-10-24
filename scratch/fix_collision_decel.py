with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """            # Signal 3: Deceleration anomaly
            decel_1 = k1.acceleration_mag < self.deceleration_threshold_ms2
            decel_2 = k2.acceleration_mag < self.deceleration_threshold_ms2
            signal_sudden_decel = bool(decel_1 or decel_2)
            severe_decel = bool(k1.acceleration_mag < -6.0 or k2.acceleration_mag < -6.0)"""

replacement = """            # Signal 3: Deceleration anomaly
            # A severe impact deceleration requires real pre-impact kinetic motion.
            # Reject Kalman filter zero-speed snapping on stationary vehicles.
            def is_real_kinetic_decel(k_obj, tid):
                if k_obj.acceleration_mag >= -6.0:
                    return False
                hist = self.track_speed_history.get(tid, [])
                if not hist:
                    return True
                recent_spds = [h[2] for h in hist[-15:]]
                max_spd = max(recent_spds) if recent_spds else k_obj.speed_kmh
                return bool(max_spd >= 8.0)

            decel_1 = k1.acceleration_mag < self.deceleration_threshold_ms2
            decel_2 = k2.acceleration_mag < self.deceleration_threshold_ms2
            severe_decel = bool(is_real_kinetic_decel(k1, t1_id) or is_real_kinetic_decel(k2, t2_id))
            signal_sudden_decel = bool(
                (decel_1 and (not self.track_speed_history.get(t1_id) or max([h[2] for h in self.track_speed_history[t1_id][-15:]]) >= 5.0))
                or (decel_2 and (not self.track_speed_history.get(t2_id) or max([h[2] for h in self.track_speed_history[t2_id][-15:]]) >= 5.0))
            )"""

text = text.replace(target, replacement)
with open('road_surveillance/events/candidates/collision.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated collision.py!')
