with open('road_surveillance/events/candidates/single_vehicle.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = """            is_shoulder_strike = bool(
                detail.get("shoulder_admitted", False)
                and (signal_severe_impact or signal_moderate_decel or signal_sudden_stop or clamped_a <= -3.0 or dh >= 20.0)
            )"""

replacement1 = """            is_shoulder_strike = bool(
                detail.get("shoulder_admitted", False)
                and (signal_severe_impact or signal_moderate_decel or signal_sudden_stop or clamped_a <= -4.0)
                and clamped_a < 0.0
                and (curr_spd <= 15.0 or signal_sudden_stop or spd_drop >= 8.0)
            )"""

target2 = """        if abs_accel >= 12.0:"""
replacement2 = """        if abs_accel >= 12.0 and clamped_a < 0.0:"""

text = text.replace(target1, replacement1)
text = text.replace(target2, replacement2)

with open('road_surveillance/events/candidates/single_vehicle.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated single_vehicle.py!')
