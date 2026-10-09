with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Check where max_recent_speed_kmh or speed_kmh or speed_drop is in collision.py
for i, l in enumerate(text.splitlines()):
    if 'max_recent_speed' in l or 'speed_drop' in l or 'speed_kmh' in l:
        print(f'{i+1:4d}: {l}')