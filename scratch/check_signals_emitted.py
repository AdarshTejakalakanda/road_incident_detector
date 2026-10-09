with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(570, 585):
    print(lines[i].strip())
