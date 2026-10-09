import os, sys
sys.path.insert(0, '.')

with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    t = f.read()

print('collision.py length;', len(t))
