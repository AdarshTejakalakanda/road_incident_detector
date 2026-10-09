import sys, os
sys.path.insert(0, os.getcwd())
from modules.vlm_arbitrator import GroqArbitrator
arb = GroqArbitrator(api_key=None, mock_fallback=True)

tel = {
    'event_type': 'collision',
    'accel_ms2': -60.0,
    'bbox_iou': 0.3521,
    'delta_v_drop': 0.0,
    'speed_kmh': 0.0,
    'max_recent_speed_kmh': 0.0,
    'S_dissip': 0.9,
    'tau_rest': 0.25,
    'post_event_rest': True,
    'track_age': 10,
    'distance_m': 1.464,
    'norm_pixel_dist': 0.87,
}
print(arb._evaluate_deterministic_high_confidence_bypass(tel, 126, 'BgdKGnFqz0I_00'))
