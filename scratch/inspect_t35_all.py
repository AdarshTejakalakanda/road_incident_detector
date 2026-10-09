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
    if 35 in tr:
        t = tr[35]
        k = kin.get(35)
        # get history from single vehicle detector
        sv = [d for d in ee.detectors if hasattr(d, 'track_history')][0]
        h = sv.track_history.get(35, [])
        if h:
            last = h[-1]
            # check if candidate was generated or why not
            cand_types = [c.event_type for c in cands if 35 in c.track_ids]
            supp_reasons = [s['reason'] for s in sv.suppressed_events if s['frame'] == f.frame_idx and 35 in s['tracks']]
            if cand_types or supp_reasons or abs(last['accel_ms2']) >= 10.0:
                print(f"F{f.frame_idx}: spd={last['speed_kmh']:.1f} a={last['accel_ms2']:.1f} age={t.track_age} cands={cand_types} supp={supp_reasons}")
