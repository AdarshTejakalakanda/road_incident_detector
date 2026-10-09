import sys
sys.path.insert(0, ".")
from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor, probe_video_metadata
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine

cfg = load_config('road_surveillance/config/default_config.yaml')
video_path = r'C:\Users\KALAKA~1\Downloads\archive\real_videos\NJvjYUBWAc0_00.mp4'
meta = probe_video_metadata(video_path)
w, h = meta.resolution
ingestor = DatasetVideoIngestor(path=video_path)
detector = YOLO26Detector(model_path='yolo26n.pt')
tracker = RobustSurveillanceTracker()
cam_cfg = CameraConfiguration.from_config_dict('cam_NJvjYUBWAc0_00', cfg.calibration.get('default'), False)
homography = PlanarHomography.from_camera_config(cam_cfg, (w, h))
lane_mgr = LaneManager.from_config(cam_cfg.lanes, homography)
geom = GeometryEngine(homography, lane_mgr, is_camera_calibrated=False)
ee = EventEngine(cfg.events, detector.names, is_camera_calibrated=False)
ee.set_resolution(w, h)

for f in ingestor.stream_video(video_path):
    b = detector.detect(f, is_low_light=False)
    tl, tr = tracker.update(b)
    kin, conv = geom.process_frame(tl, tr)
    cands = ee.process_frame(f.frame_idx, f.pts_ns, 'NJvjYUBWAc0_00', tr, kin, conv, lane_mgr)
    if cands:
        for c in cands:
            s = c.signals
            print(f"TRIGGER F{f.frame_idx}: type={c.event_type} tracks={c.track_ids} conf={c.confidence}")
            print(f"   anom_type={s.get('anomaly_type')}, s_single={s.get('composite_score')}")
            print(f"   speed={s.get('metric_speed_kmh')}, max_recent_speed={s.get('max_recent_speed_kmh')}, accel={s.get('metric_accel_ms2')}")
            print(f"   d_edge_m={s.get('d_edge_m')}, s_bound={s.get('s_bound')}, shoulder={s.get('shoulder_admitted')}")
