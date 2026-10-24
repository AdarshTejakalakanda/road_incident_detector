with open('road_surveillance/events/candidates/single_vehicle.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update compute_pre_impulse_velocity
target_pre = '''        if pre_speeds:
            v_entry = float(max(pre_speeds))
        else:
            track_spd = getattr(track, "max_recent_speed_kmh", 0.0)
            v_entry = float(max(track_spd, onset_entry.get("speed_kmh", 0.0)))'''

repl_pre = '''        if len(pre_speeds) >= 3:
            # Multi-frame consistency filter: require at least 3 consistent frames (filters 1-2 frame flicker spikes)
            rolling_mins = [min(pre_speeds[i:i + 3]) for i in range(len(pre_speeds) - 2)]
            v_entry = float(max(rolling_mins))
        elif pre_speeds:
            v_entry = float(np.median(pre_speeds))
        else:
            track_spd = getattr(track, "max_recent_speed_kmh", 0.0)
            v_entry = float(max(track_spd, onset_entry.get("speed_kmh", 0.0)))'''

assert target_pre in content, "target_pre not found"
content = content.replace(target_pre, repl_pre, 1)

# 2. Update max_recent_spd in evaluate
target_recent = '''            # Signal 1: Severe barrier impact deceleration
            max_recent_spd = max(h_entry["speed_kmh"] for h_entry in hist[:-1])'''

repl_recent = '''            # Signal 1: Severe barrier impact deceleration
            hist_speeds = [h_entry["speed_kmh"] for h_entry in hist[:-1]]
            if len(hist_speeds) >= 3:
                # Multi-frame consistency filter: require >= 3-4 consecutive frames of consistent motion
                rolling_mins = [min(hist_speeds[i:i + 3]) for i in range(len(hist_speeds) - 2)]
                max_recent_spd = float(max(rolling_mins))
            elif hist_speeds:
                max_recent_spd = float(np.median(hist_speeds))
            else:
                max_recent_spd = 0.0'''

assert target_recent in content, "target_recent not found"
content = content.replace(target_recent, repl_recent, 1)

with open('road_surveillance/events/candidates/single_vehicle.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated road_surveillance/events/candidates/single_vehicle.py with multi-frame consistency successfully")
