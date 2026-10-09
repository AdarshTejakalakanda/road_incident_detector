import json
fa = json.load(open('scratch/parsed_71_fps.json', 'r', encoding='utf-8'))

# Let us check how many of the 71 FPs have distance <= 2.0 m
dist_le2 = [x for x in fa if x['telemetry']['contact'].get('distance_m', 99.0) <= 2.0] if 'telemetry' in fa[0] else [x for x in fa if x['contact'].get('distance_m', 99.0) <= 2.0]
print('FPs with distance <= 2.0m:', len(dist_le2))
