with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    text = f.read()

for i, line in enumerate(text.splitlines()):
    if 'stationary_duration' in line:
        print(f"{i+1}: {line.strip()}")
