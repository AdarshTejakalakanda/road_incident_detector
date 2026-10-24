with open('road_surveillance/tracking/track_state.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        return float(max(speeds, default=0.0))'''

repl = '''        if len(speeds) >= 3:
            rolling_mins = [min(speeds[i:i + 3]) for i in range(len(speeds) - 2)]
            return float(max(rolling_mins, default=0.0))
        return float(np.median(speeds)) if speeds else 0.0'''

assert target in content, "target not found in track_state.py"
content = content.replace(target, repl, 1)

with open('road_surveillance/tracking/track_state.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated road_surveillance/tracking/track_state.py successfully")
