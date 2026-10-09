with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(250, 310):
    if i < len(lines):
        print(f'{i+1:4d}: {lines[i]}', end='')