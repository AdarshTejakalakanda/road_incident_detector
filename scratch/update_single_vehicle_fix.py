with open('road_surveillance/events/candidates/single_vehicle.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update default edge_margin_px to 30.0
target_edge = 'edge_margin_px: float = 20.0,'
repl_edge = 'edge_margin_px: float = 30.0,'
assert target_edge in content, "target_edge not found"
content = content.replace(target_edge, repl_edge, 1)

# 2. Update evaluate_kinematic_plausibility low v_entry
target_plaus = '''            if v_entry_kmh >= 20.0:
                # Significant kinetic energy prior to deceleration onset:
                # Rapid velocity collapse is physically consistent with a high-energy crash.
                is_plausible = True
                rejection_reason = None
                verdict = "ACCEPTED_PHYSICAL_IMPACT"
                tau_rest = (v_entry_kmh / 3.6) / (abs_accel + 1e-6)
            else:
                # Low pre-impulse speed: evaluate tau_rest against v_entry to reject stationary/Kalman jitter
                v_entry_mps = v_entry_kmh / 3.6
                tau_rest = v_entry_mps / (abs_accel + 1e-6)
                is_plausible = (tau_rest >= 0.25)
                if not is_plausible:
                    verdict = "REJECTED_JITTER"
                    rejection_reason = f"low_ventry_jitter (v_entry={v_entry_kmh:.1f}km/h, tau_rest={tau_rest:.3f}s < 0.25s)"
                else:
                    verdict = "ACCEPTED_PHYSICAL_IMPACT"
                    rejection_reason = None'''

repl_plaus = '''            if v_entry_kmh >= 15.0:
                # Significant kinetic energy prior to deceleration onset:
                # Rapid velocity collapse is physically consistent with a high-energy crash.
                is_plausible = True
                rejection_reason = None
                verdict = "ACCEPTED_PHYSICAL_IMPACT"
                tau_rest = (v_entry_kmh / 3.6) / (abs_accel + 1e-6)
            else:
                # Low pre-impulse speed: reject stationary/Kalman jitter (< 15.0 km/h cannot have high-speed crash)
                v_entry_mps = v_entry_kmh / 3.6
                tau_rest = v_entry_mps / (abs_accel + 1e-6)
                is_plausible = False
                verdict = "REJECTED_JITTER"
                rejection_reason = f"low_ventry_jitter (v_entry={v_entry_kmh:.1f}km/h < 15.0km/h)"'''

assert target_plaus in content, "target_plaus not found"
content = content.replace(target_plaus, repl_plaus, 1)

# 3. Update is_barrier_strike in evaluate
target_barrier = '''            is_barrier_strike = bool(
                (
                    (clamped_a <= self.severe_impact_decel)
                    and (curr_spd <= 6.0 or signal_sudden_stop)
                    and (spd_drop >= 10.0 or max_recent_spd >= 12.0)
                    and is_near_boundary
                    and (s_single >= 0.35 or (s_single >= 0.25 and signal_sudden_stop))
                )
                or is_shoulder_strike
            )'''

repl_barrier = '''            # Pre-existing stationary vehicle guard:
            # If track was already stationary (speed <= 5.0 km/h) over recent history,
            # it is a parked vehicle or static obstacle; it cannot crash into a barrier from rest.
            recent_speeds = [h_entry["speed_kmh"] for h_entry in hist]
            is_pre_stationary = bool(len(recent_speeds) >= 8 and float(np.median(recent_speeds[:-1])) <= 5.0)

            is_barrier_strike = bool(
                not is_pre_stationary
                and (
                    (
                        (clamped_a <= self.severe_impact_decel)
                        and (curr_spd <= 6.0 or signal_sudden_stop)
                        and (spd_drop >= 10.0 and max_recent_spd >= 15.0)
                        and is_near_boundary
                        and (s_single >= 0.35 or (s_single >= 0.25 and signal_sudden_stop))
                    )
                    or is_shoulder_strike
                )
            )'''

assert target_barrier in content, "target_barrier not found"
content = content.replace(target_barrier, repl_barrier, 1)

with open('road_surveillance/events/candidates/single_vehicle.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated road_surveillance/events/candidates/single_vehicle.py successfully")
