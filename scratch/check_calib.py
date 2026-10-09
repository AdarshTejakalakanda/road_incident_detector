import sys, os
sys.path.insert(0, os.getcwd())
from road_surveillance.config.config_loader import load_config
cfg = load_config('road_surveillance/config/default_config.yaml')
print('Calibration keys:', list(cfg.calibration.keys()))
print('BgdKGnFqz0I_00 calibrated?', 'BgdKGnFqz0I_00' in cfg.calibration)
print('kABWTDKgc9U_00 calibrated?', 'kABWTDKgc9U_00' in cfg.calibration)
