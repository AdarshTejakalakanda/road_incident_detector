import cv2
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from modules.episode_consolidator import consolidate_candidates_into_episodes
from ultralytics import YOLO

vpath = r"C:\Users\KALAKA~1\Downloads\archive\real_videos\KTGByI3N4G0_00.mp4"
cfg = load_config("road_surveillance/config/default_config.yaml")
yolo = YOLO(cfg.detection.model_path)
prep = LowLightPreprocessor(mode="N3")
vgate = VocabularyGate()
qgate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vgate)
mgate = MotionROIGate()
tracker = RobustSurveillanceTracker()
camera_cfg = CameraConfiguration.from_config_dict("KTGByI3N4G0_00", cfg.calibration.get("default"), False)
homography = PlanarHomography.from_camera_config(camera_cfg, (1280, 720))
lm = LaneManager.from_config(camera_cfg.lanes, homography)
geo = GeometryEngine(homography=homography, lane_manager=lm)
event_eng = EventEngine(config=cfg.events, is_camera_calibrated=False)
event_eng.set_resolution(1280, 720)

road_poly = vgate.get_default_roadway_polygon((720, 1280))
for d in event_eng.detectors:
    if isinstance(d, SingleVehicleAnomalyDetector):
        d.tau_single = 0.65
        d.set_roadway_polygon(road_poly)

# Configure wrong-way module in event_eng
ww_det = event_eng.wrong_way_detector
ww_mod = ww_det.module
ww_mod.flow_field.min_coherence_to_persist = 0.75
ww_mod.flow_field._cell_tids = {}

orig_observe = ww_mod.flow_field.observe_motion
def observe_with_tid(nx, ny, hvec, f_idx, tid=None):
    cell = ww_mod.flow_field._get_cell_coord(nx, ny)
    ww_mod.flow_field._cell_tids.setdefault(cell, set()).add(tid)
    orig_observe(nx, ny, hvec, f_idx)
    if len(ww_mod.flow_field._cell_tids.get(cell, set())) < 2:
        if cell in ww_mod.flow_field._persisted_models:
            del ww_mod.flow_field._persisted_models[cell]

def strict_flow(nx, ny):
    cell = ww_mod.flow_field._get_cell_coord(nx, ny)
    m = ww_mod.flow_field._persisted_models.get(cell)
    if m and m.is_persisted and len(ww_mod.flow_field._cell_tids.get(cell, set())) >= 2:
        return m
    gm = ww_mod.flow_field._global_persisted_model
    if gm and gm.is_persisted and gm.coherence >= 0.88:
        return gm
    return None

ww_mod.flow_field.get_flow_model_for_position = strict_flow

all_cands = []
ingestor = DatasetVideoIngestor(vpath)
for frame in ingestor.stream_video(vpath):
    f = frame.frame_idx
    if f > 150:
        break
    proc, _ = prep.process(frame.image)
    yres = yolo.predict(source=proc, imgsz=640, conf=0.02, device="cpu", verbose=False)[0]
    dets = [
        {"bbox": [float(x) for x in b], "conf": float(c), "class_id": int(cl), "class_name": yolo.names[int(cl)].lower()}
        for b, c, cl in zip(yres.boxes.xyxy.cpu().numpy(), yres.boxes.conf.cpu().numpy(), yres.boxes.cls.cpu().numpy())
    ] if yres.boxes is not None else []
    adm, _ = qgate.process_and_admit(dets, f, frame.pts_ns * 1e-9, proc.shape[:2], homography)
    tracklets, active = tracker.update(adm, frame.pts_ns, f)
    kin, conv = geo.process_frame(tracklets, active)
    for tid, tr in active.items():
        if tid in kin and kin[tid].speed_kmh > 12 and getattr(tr, "fused_class", "") != "candidate_vehicle_recovered":
            gc = tr.calculate_ground_contact(tr.current_bbox_xywh)
            observe_with_tid(gc[0] / 1280.0, gc[1] / 720.0, kin[tid].heading_vector, f, tid)
    cands = event_eng.process_frame(f, frame.pts_ns, "KTGByI3N4G0_00", active, kin, conv, lm)
    for c in cands:
        all_cands.append({
            "frame": f,
            "timestamp_s": round(frame.pts_ns * 1e-9, 3),
            "event_type": c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type),
            "track_ids": c.track_ids,
            "confidence": round(float(c.confidence), 2),
            "description": c.description,
            "signals": c.signals or {}
        })

print(f"Total candidates: {len(all_cands)}")
for i, c in enumerate(all_cands):
    print(f"[{i+1:2d}] F{c['frame']:3d} | {c['event_type']:24s} | Tracks: {c['track_ids']} | {c['description']}")

episodes = consolidate_candidates_into_episodes(all_cands, "KTGByI3N4G0_00", cooldown_seconds=10.0, gt_frame=91)
print(f"\nTotal Consolidated Episodes: {len(episodes)}")
for i, ep in enumerate(episodes):
    nm = getattr(ep, "attached_near_misses", [])
    print(f"Episode {i+1}: {ep.primary_event_type} F{ep.start_frame}->F{ep.end_frame} (Peak F{ep.peak_frame}) Tracks: {ep.involved_track_ids} In-GT-Window: {ep.is_in_gt_window} Near-misses: {len(nm)}")

suppressed = event_eng.get_suppressed_events()
print(f"\nTotal Suppressed Events: {len(suppressed)}")
from collections import Counter
counts = Counter(s['reason'] for s in suppressed)
for r, cnt in counts.items():
    print(f"  - {r}: {cnt}")
