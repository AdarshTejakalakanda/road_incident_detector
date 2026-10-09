with open('tests/test_vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_test2 = '''    # Test 2: Severe impact meeting deterministic high-confidence bypass -> CONFIRMED_INCIDENT
    telemetry_bypass = {
        "event_type": "collision",
        "involved_tracks": [1, 2],
        "accel_ms2": -18.5,
        "distance_m": 0.5,
        "bbox_iou": 0.38,
        "track_age": 14,
    }'''

new_test2 = '''    # Test 2: Severe impact meeting deterministic high-confidence bypass -> CONFIRMED_INCIDENT
    telemetry_bypass = {
        "event_type": "collision",
        "involved_tracks": [1, 2],
        "accel_ms2": -18.5,
        "distance_m": 0.5,
        "bbox_iou": 0.38,
        "delta_v_drop": 20.0,
        "tau_rest": 0.40,
        "track_age": 14,
    }'''

assert old_test2 in text, 'old_test2 not found'
text = text.replace(old_test2, new_test2)

with open('tests/test_vlm_arbitrator.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated test_vlm_arbitrator successfully!')
