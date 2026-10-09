import json

with open("scratch/z4_diagnostic_output.json") as f:
    data = json.load(f)

print("=== TRACKS 7, 10, 14, 21, 22 TRAJECTORIES ===")
# We have tracks recorded in frame_tracks_info for 150..180
# Let's inspect F150 to F180
for f in range(150, 181):
    f_str = str(f)
    tracks = data["tracks_f150_f180"].get(f_str, [])
    t_map = {t["id"]: t for t in tracks}
    
    t7 = t_map.get(7)
    t10 = t_map.get(10)
    t14 = t_map.get(14)
    t21 = t_map.get(21)
    t22 = t_map.get(22)
    
    line = f"F{f:3d}: "
    if t10:
        line += f"T10(box={t10['bbox']}, spd={t10['speed_kmh']}, hdg={t10['heading_deg']}, acc={t10['accel_ms2']}) "
    if t22:
        line += f"T22(box={t22['bbox']}, spd={t22['speed_kmh']}) "
    if t14:
        line += f"T14(box={t14['bbox']}, spd={t14['speed_kmh']}) "
    if t7:
        line += f"T7(box={t7['bbox']}) "
    print(line)
