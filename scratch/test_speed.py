import sys
sys.path.insert(0, ".")
from road_surveillance.config import load_config
from road_surveillance.video import DatasetVideoIngestor, probe_video_metadata
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
import numpy as np

def compute_track_max_recent_speed(tr, window_frames=15, min_age=4):
    if len(tr.position_history) < min_age or len(tr.timestamps) < min_age:
        return 0.0
    
    # Calculate speeds over smoothed step (e.g. 3 frames step) to eliminate single-pixel jitter
    # ignoring the unconfirmed initial frames (< min_age)
    speeds = []
    w = tr.current_bbox_xywh[2]
    h = tr.current_bbox_xywh[3]
    char_dim = max(15.0, max(w, h))
    scale = 4.5 / char_dim

    # Look back up to window_frames
    start_idx = max(min_age - 1, len(tr.position_history) - window_frames)
    end_idx = len(tr.position_history)

    # Use a 3-frame step displacement: pos[i] - pos[i-3]
    step = 3
    for i in range(start_idx + step, end_idx):
        dt = (tr.timestamps[i] - tr.timestamps[i - step]) * 1e-9
        if dt > 1e-4:
            p1 = tr.position_history[i]
            p0 = tr.position_history[i - step]
            disp = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            spd_kmh = (disp / dt) * scale * 3.6
            speeds.append(spd_kmh)
    
    return max(speeds, default=0.0)

video_path = r'C:\Users\KALAKA~1\Downloads\archive\real_videos\NJvjYUBWAc0_00.mp4'
ingestor = DatasetVideoIngestor(path=video_path)
detector = YOLO26Detector(model_path='yolo26n.pt')
tracker = RobustSurveillanceTracker()

for f in ingestor.stream_video(video_path):
    b = detector.detect(f, is_low_light=False)
    tl, tr = tracker.update(b)
    if f.frame_idx in [107, 320, 365]:
        for tid in [17, 35]:
            if tid in tr:
                spd = compute_track_max_recent_speed(tr[tid])
                print(f"F{f.frame_idx} Track {tid}: computed max_recent_speed = {spd:.1f} km/h")
