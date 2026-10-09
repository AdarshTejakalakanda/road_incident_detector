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
    if f.frame_idx < 270:
        b = detector.detect(f, is_low_light=False)
        tl, tr = tracker.update(b)
        kin, conv = geom.process_frame(tl, tr)
        continue
    b = detector.detect(f, is_low_light=False)
    tl, tr = tracker.update(b)
    kin, conv = geom.process_frame(tl, tr)
    cands = ee.process_frame(f.frame_idx, f.pts_ns, 'NJvjYUBWAc0_00', tr, kin, conv, lane_mgr)
    if 35 in tr:
        sv_det = [d for d in ee.detectors if hasattr(d, 'track_history')][0]
        h = sv_det.track_history.get(35, [])
        if h:
            last = h[-1]
            max_spd = max([x['speed_kmh'] for x in h[:-1]], default=0.0) if len(h) > 1 else 0.0
            print(f"F{f.frame_idx}: spd={last['speed_kmh']:.1f}, max_spd={max_spd:.1f}, a={last['accel_ms2']:.1f}, tr_age={tr[35].track_age}")
            for c in cands:
                if 35 in c.track_ids:
                    print(f"   --> CAND: {c.event_type} sig={c.signals.get('anomaly_type')}")
