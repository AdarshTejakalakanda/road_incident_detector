"""CLI Entrypoint to Run the Updated Road Surveillance Pipeline (Iteration N).

Usage Examples:
    # Run on a video file path:
    python run_pipeline.py --video "C:/path/to/video.mp4"

    # Run on a video ID from the dataset archive:
    python run_pipeline.py --video "yzNEliTlFGk_00"

    # Run with custom frame limit and evidence output:
    python run_pipeline.py --video "nAXTthLfgtI_00" --max-frames 200 --save-evidence
"""

import os
import sys
import time
import collections
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import cv2
import numpy as np
import torch

torch.set_num_threads(8)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

from road_surveillance.config import load_config
from road_surveillance.calibration.camera_config import CameraConfiguration
from road_surveillance.calibration.homography import PlanarHomography
from road_surveillance.calibration.lane_manager import LaneManager
from road_surveillance.geometry.engine import GeometryEngine
from road_surveillance.events.engine import EventEngine
from road_surveillance.events.candidates.single_vehicle import SingleVehicleAnomalyDetector
from road_surveillance.tracking.surveillance_tracker import RobustSurveillanceTracker
from road_surveillance.video.ingestion import DatasetVideoIngestor
from road_surveillance.preprocessing.illumination import LowLightPreprocessor
from road_surveillance.visualization import FrameAnnotator, PlaybackRenderer

from modules.detector_quality_gate import DetectorQualityGate
from modules.vocabulary_gate import VocabularyGate
from modules.motion_roi_gate import MotionROIGate
from modules.vlm_arbitrator import create_evidence_strip, GroqArbitrator
from modules.episode_consolidator import consolidate_candidates_into_episodes, ActorDispatchTracker
from modules.scene_escalator import (
    SceneEscalator,
    SceneEscalatorDisposition,
    EpisodeStatus,
    SceneEscalationResult,
    CandidateProposal,
    TrackState,
    TrackerContext,
    StreamMetadata,
)
from modules.telemetry_accumulator import TelemetryAccumulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_pipeline")


def resolve_video_path(video_input: str) -> str:
    """Resolve file path from direct path or dataset archive."""
    if os.path.isfile(video_input):
        return os.path.abspath(video_input)

    vid_base = os.path.basename(video_input)
    if not vid_base.endswith(".mp4"):
        vid_base = f"{vid_base}.mp4"
    candidates = [
        video_input,
        os.path.join(r"C:\Users\KALAKA~1\Downloads\archive\real_videos", vid_base),
        os.path.join(os.path.expanduser("~"), "Downloads", "archive", "real_videos", vid_base),
        os.path.join("data", "videos", vid_base),
        os.path.join("real_videos", vid_base),
        os.path.join(".", vid_base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    raise FileNotFoundError(f"Video '{video_input}' could not be located on disk.")


def compute_union_roi(
    boxes: List[Tuple[float, float, float, float]],
    frame_shape: Tuple[int, int],
    margin_factor: float = 0.5,
    min_size: int = 220,
) -> Tuple[int, int, int, int]:
    h_img, w_img = frame_shape[:2]
    if not boxes:
        cx, cy = w_img // 2, h_img // 2
        return (
            max(0, cx - min_size // 2),
            max(0, cy - min_size // 2),
            min(w_img, cx + min_size // 2),
            min(h_img, cy + min_size // 2),
        )

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    bw = max(10.0, x2 - x1)
    bh = max(10.0, y2 - y1)

    pad_w = max((min_size - bw) / 2.0, bw * margin_factor)
    pad_h = max((min_size - bh) / 2.0, bh * margin_factor)

    return (
        int(max(0, np.floor(x1 - pad_w))),
        int(max(0, np.floor(y1 - pad_h))),
        int(min(w_img, np.ceil(x2 + pad_w))),
        int(min(h_img, np.ceil(y2 + pad_h))),
    )


def load_frame_at(video_path: str, target_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_idx - 1))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def get_frame_image(target_f: int, buffer: Any, video_path: str) -> np.ndarray:
    """Retrieves frame from in-memory ring buffer if available, else seeks from disk."""
    if buffer is not None:
        for f, im in buffer:
            if f == target_f:
                return im.copy()
    return load_frame_at(video_path, target_f)


def extract_actor_crop_boxes(
    target_frame: int,
    track_ids: List[int],
    rx1: int,
    ry1: int,
    crop_w: int,
    crop_h: int,
    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]],
    search_radius: int = 5,
) -> Dict[int, Tuple[int, int, int, int]]:
    """Extracts bounding boxes of actors for a given frame in crop-relative coordinates."""
    boxes_crop: Dict[int, Tuple[int, int, int, int]] = {}
    for tid in track_ids:
        found_box = None
        if target_frame in frame_tracks_cache and tid in frame_tracks_cache[target_frame]:
            found_box = frame_tracks_cache[target_frame][tid]
        else:
            for offset in range(1, search_radius + 1):
                for f_chk in (target_frame - offset, target_frame + offset):
                    if f_chk in frame_tracks_cache and tid in frame_tracks_cache[f_chk]:
                        found_box = frame_tracks_cache[f_chk][tid]
                        break
                if found_box is not None:
                    break
        if found_box is not None:
            bx1, by1, bx2, by2 = found_box
            cx1 = max(0, min(crop_w - 1, int(round(bx1 - rx1))))
            cy1 = max(0, min(crop_h - 1, int(round(by1 - ry1))))
            cx2 = max(0, min(crop_w - 1, int(round(bx2 - rx1))))
            cy2 = max(0, min(crop_h - 1, int(round(by2 - ry1))))
            if cx2 > cx1 and cy2 > cy1:
                boxes_crop[tid] = (cx1, cy1, cx2, cy2)
    return boxes_crop


def build_triptych_evidence_card(
    crop_pre: np.ndarray,
    crop_impact: np.ndarray,
    crop_post: np.ndarray,
    video_id: str,
    candidate_frame: int,
    timestamp_s: float,
    event_type: str,
    track_ids: List[int],
    actor_boxes_pre: Dict[int, Tuple[int, int, int, int]],
    actor_boxes_impact: Dict[int, Tuple[int, int, int, int]],
    actor_boxes_post: Dict[int, Tuple[int, int, int, int]],
    timeline_meta: Dict[str, Any],
    telemetry: Dict[str, Any],
    target_h: int = 340,
) -> np.ndarray:
    """Builds a unified 3-Panel Fixed Collision Zone Triptych Evidence Card.

    Option B: [PRE: T-5s | TRIGGER: T | POST: T+5s]
    With individual panel bounding boxes, stage sub-banners, and master top HUD banner.
    """
    crops_in = [crop_pre, crop_impact, crop_post]
    boxes_in = [actor_boxes_pre, actor_boxes_impact, actor_boxes_post]

    f_pre = timeline_meta.get("f_pre", max(1, candidate_frame - 150))
    f_imp = candidate_frame
    f_post = timeline_meta.get("f_post", candidate_frame + 150)
    dt_pre = float(timeline_meta.get("dt_pre_s", 5.0))
    dt_post = float(timeline_meta.get("dt_post_s", 5.0))

    sub_titles = [
        f"1: PRE-EVENT (T-{dt_pre:.1f}s | F{f_pre:03d})",
        f"2: TRIGGER / IMPACT (T | F{f_imp:03d})",
        f"3: POST-EVENT AFTERMATH (T+{dt_post:.1f}s | F{f_post:03d})",
    ]
    stage_tags = ["APPROACH", "TRIGGER", "AFTERMATH"]
    panel_colors = [
        (0, 220, 100),   # Green for approach
        (0, 0, 255),     # Red for impact
        (0, 215, 255),   # Amber/yellow for aftermath
    ]

    palette = [(0, 220, 100), (0, 165, 255), (255, 128, 0), (200, 200, 255)]
    rendered_panels = []
    target_panel_h = max(180, target_h - 30)

    for i in range(3):
        cr = crops_in[i]
        if cr is None or cr.size == 0:
            cr = np.zeros((target_panel_h, target_panel_h, 3), dtype=np.uint8)
        ch, cw = cr.shape[:2]
        if ch <= 0 or cw <= 0:
            cr = np.zeros((target_panel_h, target_panel_h, 3), dtype=np.uint8)
            ch, cw = target_panel_h, target_panel_h

        canvas = cr.copy()
        cur_boxes = boxes_in[i]
        for p_idx, tid in enumerate(track_ids):
            box = cur_boxes.get(tid)
            if box:
                bx1, by1, bx2, by2 = box
                color = panel_colors[i] if len(track_ids) == 1 else palette[p_idx % len(palette)]
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
                lbl = f"ID:{tid} {stage_tags[i]}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tag_y1 = max(0, by1 - th - 6)
                cv2.rectangle(canvas, (bx1, tag_y1), (min(cw, bx1 + tw + 6), by1), (20, 20, 20), -1)
                cv2.putText(canvas, lbl, (bx1 + 3, by1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        scale = target_panel_h / float(ch)
        new_w = max(10, int(round(cw * scale)))
        resized = cv2.resize(canvas, (new_w, target_panel_h), interpolation=cv2.INTER_AREA)

        sub_banner = np.zeros((30, new_w, 3), dtype=np.uint8)
        cv2.rectangle(sub_banner, (0, 0), (new_w, 30), (25, 25, 30), -1)
        cv2.rectangle(sub_banner, (0, 0), (new_w, 2), panel_colors[i], -1)
        cv2.putText(sub_banner, sub_titles[i], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        panel = np.vstack([sub_banner, resized])
        rendered_panels.append(panel)

    divider_h = rendered_panels[0].shape[0]
    divider = np.full((divider_h, 3, 3), (100, 100, 110), dtype=np.uint8)
    strip_elements = []
    for i, p in enumerate(rendered_panels):
        strip_elements.append(p)
        if i < len(rendered_panels) - 1:
            strip_elements.append(divider)
    panel_strip = np.hstack(strip_elements)

    strip_w = panel_strip.shape[1]
    banner_h = 60
    master_banner = np.zeros((banner_h, strip_w, 3), dtype=np.uint8)
    cv2.rectangle(master_banner, (0, 0), (strip_w, banner_h), (18, 18, 22), -1)
    cv2.rectangle(master_banner, (0, 0), (strip_w, 3), (0, 165, 255), -1)

    title = f"[INCIDENT EVIDENCE: {event_type.upper()}]  Trigger: F{candidate_frame:03d} ({timestamp_s:.2f}s)  Clip: {video_id}"
    cv2.putText(master_banner, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    spd = float(telemetry.get("speed_kmh", 0.0))
    accel = float(telemetry.get("accel_ms2", 0.0))
    tau_rest = float(telemetry.get("tau_rest", 0.0))
    dist = float(telemetry.get("distance_m", 0.0))
    conf = float(telemetry.get("confidence", 0.0))
    metrics_str = f"Actors: {track_ids} | Speed: {spd:.1f} km/h | Accel: {accel:+.1f} m/s2 | Rest: {tau_rest:.1f}s | Dist: {dist:.1f}m | Conf: {conf:.2f}"
    cv2.putText(master_banner, metrics_str, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([master_banner, panel_strip])


def build_incident_roi_card(
    crop_img: np.ndarray,
    video_id: str,
    candidate_frame: int,
    timestamp_s: float,
    event_type: str,
    track_ids: List[int],
    actor_boxes_crop: Dict[int, Tuple[int, int, int, int]],
    telemetry: Dict[str, Any],
) -> np.ndarray:
    """Backward-compatible single-panel ROI card wrapper."""
    return build_triptych_evidence_card(
        crop_pre=crop_img,
        crop_impact=crop_img,
        crop_post=crop_img,
        video_id=video_id,
        candidate_frame=candidate_frame,
        timestamp_s=timestamp_s,
        event_type=event_type,
        track_ids=track_ids,
        actor_boxes_pre=actor_boxes_crop,
        actor_boxes_impact=actor_boxes_crop,
        actor_boxes_post=actor_boxes_crop,
        timeline_meta={"f_pre": candidate_frame, "f_post": candidate_frame, "dt_pre_s": 0.0, "dt_post_s": 0.0},
        telemetry=telemetry,
    )


def run_pipeline(
    video_path: str,
    config_path: str = "road_surveillance/config/default_config.yaml",
    max_frames: Optional[int] = None,
    save_evidence: bool = True,
    save_video: bool = True,
    output_video: Optional[str] = None,
    output_json: Optional[str] = None,
    imgsz: Optional[Union[int, str]] = None,
) -> Dict[str, Any]:
    resolved_path = resolve_video_path(video_path)
    video_id = os.path.splitext(os.path.basename(resolved_path))[0]

    print("\n" + "=" * 80)
    print(f"RUNNING UPDATED ROAD SURVEILLANCE PIPELINE (ITERATION N)")
    print(f"Video: {resolved_path}")
    print(f"ID:    {video_id}")
    print("=" * 80)

    cfg = load_config(config_path)
    yolo_model = YOLO(cfg.detection.model_path)
    preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
    vocab_gate = VocabularyGate()
    quality_gate = DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
    motion_gate = MotionROIGate(
        min_cluster_area=250,
        max_cluster_area=3500,
        persistence_threshold=4,
        crop_resolution=320,
        crop_conf_threshold=0.05,
        max_crops_per_frame=1,
        min_crop_interval=2,
    )
    arbitrator = GroqArbitrator(mock_fallback=True)

    cap_meta = cv2.VideoCapture(resolved_path)
    vid_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    vid_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    stream_fps = float(cap_meta.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap_meta.release()

    stream_metadata = StreamMetadata(
        fps=stream_fps,
        duration_sec=(total_frames / stream_fps) if total_frames > 0 and stream_fps > 0 else 0.0,
        total_frames=total_frames,
        width=vid_w,
        height=vid_h,
    )
    tracker_context = TrackerContext(fps=stream_fps)
    telemetry_acc = TelemetryAccumulator()

    is_calibrated = bool(cfg.calibration.get(video_id))
    raw_calib = cfg.calibration.get(video_id) or cfg.calibration.get("default")
    camera_cfg = CameraConfiguration.from_config_dict(
        camera_id=video_id,
        cfg_obj=raw_calib,
        is_calibrated=is_calibrated,
    )
    homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(vid_w, vid_h))
    lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)
    geometry_engine = GeometryEngine(
        homography=homography,
        lane_manager=lane_manager,
        is_camera_calibrated=camera_cfg.is_calibrated,
    )
    event_engine = EventEngine(
        config=cfg.events,
        is_camera_calibrated=camera_cfg.is_calibrated,
    )
    event_engine.set_resolution(vid_w, vid_h)

    road_poly = vocab_gate.get_default_roadway_polygon((vid_h, vid_w))
    for d in event_engine.detectors:
        if isinstance(d, SingleVehicleAnomalyDetector):
            d.tau_single = 0.65
            d.set_roadway_polygon(road_poly)

    tracker = RobustSurveillanceTracker()
    motion_gate.reset()

    annotator = FrameAnnotator(homography=homography, lane_manager=lane_manager) if save_video else None
    renderer = PlaybackRenderer(
        display=False,
        save_processed_video=save_video,
        output_dir="output/videos",
    ) if save_video else None
    if renderer and save_video:
        renderer.setup_video_writer(video_id, stream_fps, vid_w, vid_h, custom_path=output_video)

    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    all_candidates: List[Dict[str, Any]] = []
    crops_evaluated = 0
    micro_recovered = 0
    admitted_candidates = 0

    # Rolling ring buffer for pre-event retrieval (e.g. 6.5 seconds @ stream_fps)
    ring_buf_len = max(60, int(round(stream_fps * 6.5)))
    frame_ring_buffer: collections.deque = collections.deque(maxlen=ring_buf_len)

    ingestor = DatasetVideoIngestor(path=resolved_path)
    t0 = time.perf_counter()
    last_f_idx = 0

    print("Processing video stream...")
    for frame in ingestor.stream_video(resolved_path):
        f_idx = frame.frame_idx
        if max_frames and f_idx > max_frames:
            break
        last_f_idx = f_idx
        img = frame.image
        frame_ring_buffer.append((f_idx, img.copy()))
        proc_img, _ = preprocessor.process(img)
        h, w = proc_img.shape[:2]
        pts_ns = frame.pts_ns
        timestamp_s = float(pts_ns * 1e-9)

        # 1. Primary YOLO Native Feed Inference (preserves native video feed details)
        if imgsz is not None and str(imgsz).lower() not in ("native", "none", "0"):
            det_imgsz = int(imgsz) if str(imgsz).isdigit() else 640
        elif cfg.detection.input_resolution not in (None, "native", "none", "0"):
            det_imgsz = int(cfg.detection.input_resolution) if str(cfg.detection.input_resolution).isdigit() else 640
        else:
            native_stride = 32
            det_imgsz = (
                int(np.ceil(h / float(native_stride)) * native_stride),
                int(np.ceil(w / float(native_stride)) * native_stride),
            )

        yolo_res = yolo_model.predict(source=proc_img, imgsz=det_imgsz, conf=0.02, device="cpu", verbose=False)[0]
        raw_dets = []
        if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
            b_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
            b_conf = yolo_res.boxes.conf.cpu().numpy()
            b_cls = yolo_res.boxes.cls.cpu().numpy()
            for bx, bc, bcl in zip(b_xyxy, b_conf, b_cls):
                raw_dets.append({
                    "bbox": [float(x) for x in bx],
                    "conf": float(bc),
                    "class_id": int(bcl),
                    "class_name": yolo_model.names[int(bcl)].lower(),
                })

        # 2. Motion-Guided ROI Zoom Pass (Patch N.2)
        recovered_micro, m_tel = motion_gate.process_and_recover(
            frame_curr=img,
            frame_idx=f_idx,
            primary_raw_detections=raw_dets,
            roadway_polygon=road_poly,
            yolo_model=yolo_model,
        )
        raw_dets.extend(recovered_micro)
        crops_evaluated += m_tel["roi_crops_evaluated"]
        micro_recovered += m_tel["recovered_micro_vehicles"]

        # 3. Detector Quality Gate (Patch N.1 semantic recovery)
        admitted, qtel = quality_gate.process_and_admit(
            raw_detections=raw_dets,
            frame_idx=f_idx,
            timestamp_s=timestamp_s,
            frame_shape=(h, w),
            homography=homography,
            roadway_polygon=road_poly,
        )
        admitted_candidates += qtel.get("candidate_recovered_admitted", 0)

        # 4. Robust Tracker
        tracklets, active_tracks = tracker.update(admitted, pts_ns=pts_ns, frame_idx=f_idx)

        frame_tracks_cache[f_idx] = {
            tid: (
                tr.current_bbox_xywh[0],
                tr.current_bbox_xywh[1],
                tr.current_bbox_xywh[0] + tr.current_bbox_xywh[2],
                tr.current_bbox_xywh[1] + tr.current_bbox_xywh[3],
            )
            for tid, tr in active_tracks.items()
        }

        # 5. Kinematics & Events
        kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)
        telemetry_acc.record_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            timestamp_s=timestamp_s,
            active_tracks=active_tracks,
            kinematics_map=kinematics,
        )
        cands = event_engine.process_frame(
            frame_idx=f_idx,
            pts_ns=pts_ns,
            video_id=video_id,
            tracks=active_tracks,
            kinematics=kinematics,
            convergences=convergences,
            lane_manager=lane_manager,
        )

        for c in cands:
            etype = c.event_type.value if hasattr(c.event_type, "value") else str(c.event_type)
            all_candidates.append({
                "frame": f_idx,
                "timestamp_s": round(timestamp_s, 3),
                "event_type": etype,
                "track_ids": c.track_ids,
                "confidence": round(float(c.confidence), 2),
                "description": c.description,
                "signals": c.signals or {},
            })
            print(f"  [ALERT F{f_idx:03d} | {timestamp_s:.2f}s] {etype.upper()} (conf={c.confidence:.2f}, tracks={c.track_ids})")

        # 6. Render & save annotated video frame
        if renderer is not None and annotator is not None and renderer.writer is not None:
            annotated_frame = annotator.annotate(
                image=img,
                tracklets=tracklets,
                tracks=active_tracks,
                kinematics=kinematics,
                active_candidates=cands,
                fps=stream_fps,
                frame_idx=f_idx,
                pts_ns=pts_ns,
            )
            renderer.render(annotated_frame, stream_fps)

    if renderer is not None:
        renderer.close()
        if save_video and renderer.last_output_path:
            print(f"\n[INFO] Annotated video saved to: {renderer.last_output_path}")

    elapsed_s = time.perf_counter() - t0
    fps = round(last_f_idx / elapsed_s, 1) if elapsed_s > 0 else 0.0

    # 6. Episode Consolidation & Video-VLM Arbitration (Iteration P.3)
    episodes = consolidate_candidates_into_episodes(
        raw_candidates=all_candidates,
        video_id=video_id,
        cooldown_seconds=10.0,
        max_gap_frames=60,
    )

    if stream_metadata.duration_sec <= 0.0 or (max_frames and last_f_idx < stream_metadata.total_frames):
        stream_metadata.duration_sec = last_f_idx / stream_metadata.fps

    # --- PHASE 2 SCENE ESCALATOR INTEGRATION ---
    print(f"\nEvaluating {len(episodes)} consolidated episodes with SceneEscalator (tau_aftermath=0.50)...")
    escalated_episodes = []
    quarantined_records = []
    all_aftermath_audit_records = []
    scene_escalator = SceneEscalator(tau_aftermath=0.50)

    # Reconcile stream end timestamp with synthetic timebase
    stream_duration_s = float(total_frames / stream_fps) if (total_frames > 0 and stream_fps > 0) else 0.0
    if max_frames and last_f_idx < total_frames and stream_fps > 0:
        t_stream_end = float(last_f_idx / stream_fps)
    else:
        t_stream_end = float(total_frames / stream_fps) if stream_fps > 0 else stream_duration_s

    for ep in episodes:
        t_anchor = float(ep.peak_candidate["timestamp_s"])
        delta_t_avail = t_stream_end - t_anchor

        if delta_t_avail < 0.75:
            res = SceneEscalationResult(
                disposition=SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW,
                score=None,
                window_state="INSUFFICIENT_AFTERMATH_WINDOW",
                telemetry={"delta_t_avail": delta_t_avail},
            )
        else:
            t_aftermath_eval = min(5.00, delta_t_avail)
            t_end = t_anchor + t_aftermath_eval
            window_history = telemetry_acc.get_window(t_start=t_anchor, t_end=t_end)

            res = scene_escalator.evaluate_candidate(
                candidate=ep.peak_candidate,
                track_history=window_history,
                fps=stream_fps,
                horizon_seconds=t_aftermath_eval,
            )
            res.window_state = (
                "COMPLETE_AFTERMATH_WINDOW" if delta_t_avail >= 4.50
                else "PARTIAL_AFTERMATH_WINDOW"
            )

        ep.aftermath_result = res

        # Safely extract actual fields from SceneEscalationResult
        raw_telemetry = getattr(res, "features", getattr(res, "telemetry", {}))
        normalized_channels = getattr(res, "channel_scores", getattr(res, "channels", {}))
        raw_window_state = getattr(res, "window_state", "UNKNOWN")
        raw_disposition = getattr(res, "disposition", "UNKNOWN")
        res_score = getattr(res, "score", getattr(res, "aftermath_score", None))

        unified_audit_entry = {
            "episode_id": ep.episode_id,
            "anchor_frame": ep.peak_candidate["frame"],
            "canonical_type": getattr(res, "canonical_type", "UNKNOWN"),
            "disposition": raw_disposition.value if hasattr(raw_disposition, "value") else str(raw_disposition),
            "score": round(float(res_score), 4) if (res_score is not None and np.isfinite(res_score)) else None,
            "window_state": raw_window_state.value if hasattr(raw_window_state, "value") else str(raw_window_state),
            "telemetry": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (None if isinstance(v, float) and not np.isfinite(v) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in raw_telemetry.items()},
            "channels": {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else (None if isinstance(v, float) and not np.isfinite(v) else (int(v) if isinstance(v, (int, np.integer)) else v))) for k, v in normalized_channels.items()},
            "safeguards": getattr(res, "safeguards", {}),
        }
        all_aftermath_audit_records.append(unified_audit_entry)

        # Backward-compatible suppression record format
        suppression_entry = {
            "episode_id": ep.episode_id,
            "anchor_frame": ep.peak_candidate["frame"],
            "disposition": res.disposition.value if hasattr(res.disposition, "value") else str(res.disposition),
            "score": res.score,
            "window_state": res.window_state,
            "telemetry": res.telemetry,
            "channels": res.channels,
        }

        if res.disposition == SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT:
            ep.aftermath_telemetry = res.telemetry
            ep.aftermath_score = res.score
            ep.aftermath_type = res.canonical_type
            escalated_episodes.append(ep)
            print(f"  [ESCALATED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> {res.disposition.value} (score={res.score}, type={res.canonical_type})")
        elif res.disposition == SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW:
            cand_spd = float(ep.peak_candidate.get("signals", {}).get("metric_speed_kmh", ep.peak_candidate.get("speed_kmh", 0.0)))
            cand_conf = float(ep.peak_candidate.get("confidence", 0.0))
            if cand_spd > 8.0 and cand_conf >= 0.70:
                escalated_episodes.append(ep)
                print(f"  [ESCALATED FALLBACK] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW (spd={cand_spd:.1f} km/h)")
            else:
                ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
                quarantined_records.append(suppression_entry)
                print(f"  [SUPPRESSED FALLBACK] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (stationary/low-confidence boundary noise)")
        else:
            ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
            quarantined_records.append(suppression_entry)
            print(f"  [SUPPRESSED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (score={res.score}, type={res.canonical_type})")

    # Write unified run artifact containing 100% of evaluated episodes
    os.makedirs("artifacts", exist_ok=True)
    audit_path_all = f"artifacts/aftermath_evaluation_audit_{video_id}.json"
    with open(audit_path_all, "w", encoding="utf-8") as f:
        json.dump(all_aftermath_audit_records, f, indent=2)
    print(f"Unified aftermath audit saved: {audit_path_all} ({len(all_aftermath_audit_records)} evaluated records)")

    # Preserve backward compatibility: write suppression audit as well
    audit_path_suppressed = f"artifacts/aftermath_suppression_audit_{video_id}.json"
    with open(audit_path_suppressed, "w", encoding="utf-8") as f:
        json.dump(quarantined_records, f, indent=2)
    print(f"Quarantine audit saved: {audit_path_suppressed} ({len(quarantined_records)} suppressed records)")

    # Pass ONLY escalated episodes to downstream Gate 4 arbitration
    episodes = escalated_episodes

    print(f"\nCompleted {last_f_idx} frames in {elapsed_s:.2f}s ({fps} FPS).")
    print(f"Progression: {len(all_candidates)} Raw Proposals -> {len(episodes)} Escalated Episodes.")
    print(f"Arbitrating {len(episodes)} episodes via Video-VLM Arbitration Layer...")

    arbitrated_incidents = []
    dispatch_tracker = ActorDispatchTracker(cooldown_seconds=10.0)
    os.makedirs("artifacts", exist_ok=True)
    # Save individual run separately in subfolder named with video_id under roi_crops
    roi_dir = os.path.join(PROJECT_ROOT, "output", "roi_crops", video_id)
    os.makedirs(roi_dir, exist_ok=True)

    for idx, ep in enumerate(episodes, 1):
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_t = cand.get("timestamp_s", cand_f / stream_fps)

        # 5-second temporal timeline gap (T-5s Approach | T Trigger | T+5s Post-Aftermath)
        dt_frames = int(round(5.0 * stream_fps))
        f_app = max(1, cand_f - dt_frames)
        f_imp = cand_f
        f_post = min(last_f_idx, cand_f + dt_frames)
        dt_pre_s = round((f_imp - f_app) / stream_fps, 2)
        dt_post_s = round((f_post - f_imp) / stream_fps, 2)

        img_app = get_frame_image(f_app, frame_ring_buffer, resolved_path)
        img_imp = get_frame_image(f_imp, frame_ring_buffer, resolved_path)
        img_post = get_frame_image(f_post, frame_ring_buffer, resolved_path)

        # Extract candidate telemetry and involved actor track IDs
        cand_signals = cand.get("signals", {})
        cand_type = cand.get("event_type", "incident")
        clean_type = cand_type.lower().replace(" ", "_").replace("/", "_")
        conf_val = float(cand.get("confidence", 0.0))
        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -4.0)))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        spd_val = float(cand_signals.get("metric_speed_kmh", cand_signals.get("speed_kmh", 0.0)))
        tids = cand.get("track_ids", [])

        # Option B: Fixed Unified Spatial ROI covering approach, impact, and post-rest aftermath
        boxes = []
        sample_frames = set()
        for base_f in [f_app, f_imp, f_post]:
            for offset in range(-5, 6):
                f_chk = base_f + offset
                if 1 <= f_chk <= last_f_idx:
                    sample_frames.add(f_chk)
        for sf in range(f_app, f_post + 1, 5):
            sample_frames.add(sf)

        for f in sorted(sample_frames):
            if f in frame_tracks_cache:
                for tid in tids:
                    if tid in frame_tracks_cache[f]:
                        boxes.append(frame_tracks_cache[f][tid])

        if not boxes:
            for f in range(max(1, f_imp - 5), min(last_f_idx + 1, f_imp + 6)):
                if f in frame_tracks_cache:
                    for tid in tids:
                        if tid in frame_tracks_cache[f]:
                            boxes.append(frame_tracks_cache[f][tid])

        if not boxes:
            px_c = cand_signals.get("pixel_centroid")
            if px_c and len(px_c) == 2:
                cx, cy = px_c
            else:
                cx, cy = vid_w / 2.0, vid_h / 2.0
            boxes = [(cx - 100, cy - 100, cx + 100, cy + 100)]

        roi = compute_union_roi(boxes, (vid_h, vid_w), margin_factor=0.40, min_size=320)
        rx1, ry1, rx2, ry2 = roi

        crop_pre = img_app[ry1:ry2, rx1:rx2].copy()
        crop_impact = img_imp[ry1:ry2, rx1:rx2].copy()
        crop_post = img_post[ry1:ry2, rx1:rx2].copy()

        # Build actor boxes relative to crop coordinates for all 3 stages
        crop_h, crop_w = crop_impact.shape[:2]
        actor_boxes_pre = extract_actor_crop_boxes(f_app, tids, rx1, ry1, crop_w, crop_h, frame_tracks_cache)
        actor_boxes_imp = extract_actor_crop_boxes(f_imp, tids, rx1, ry1, crop_w, crop_h, frame_tracks_cache)
        actor_boxes_post = extract_actor_crop_boxes(f_post, tids, rx1, ry1, crop_w, crop_h, frame_tracks_cache)

        aftermath_res = getattr(ep, "aftermath_result", None)
        tau_rest_val = 0.0
        if aftermath_res is not None:
            tau_rest_val = float(getattr(aftermath_res, "telemetry", {}).get("tau_rest", 0.0))

        meta_dict = {
            "speed_kmh": spd_val,
            "accel_ms2": acc_val,
            "distance_m": dist_val,
            "confidence": conf_val,
            "tau_rest": tau_rest_val,
        }
        timeline_meta = {
            "f_pre": f_app,
            "f_imp": f_imp,
            "f_post": f_post,
            "dt_pre_s": dt_pre_s,
            "dt_post_s": dt_post_s,
        }

        roi_card = build_triptych_evidence_card(
            crop_pre=crop_pre,
            crop_impact=crop_impact,
            crop_post=crop_post,
            video_id=video_id,
            candidate_frame=cand_f,
            timestamp_s=cand_t,
            event_type=cand_type,
            track_ids=tids,
            actor_boxes_pre=actor_boxes_pre,
            actor_boxes_impact=actor_boxes_imp,
            actor_boxes_post=actor_boxes_post,
            timeline_meta=timeline_meta,
            telemetry=meta_dict,
        )

        strip = roi_card

        if save_evidence:
            strip_path = f"artifacts/{video_id}_ep_{idx:02d}_f{cand_f:03d}_evidence.jpg"
            cv2.imwrite(strip_path, strip)

        # Task S.A.3: Actor-level dispatch cooldown check
        should_suppress, cd_log = dispatch_tracker.should_suppress_dispatch(cand.get("track_ids", []), cand_t)
        if should_suppress:
            ep.dispatch_suppressed = True
            ep.suppression_reason = "actor_cooldown_active"
            ep.cooldown_telemetry = cd_log
            print(f"  [{idx}/{len(episodes)}] Episode {ep.episode_id} @ F{cand_f:03d} -> SUPPRESSED (actor_cooldown_active: {cd_log['actor_ids']} dt={cd_log['delta_time_s']}s)")
            continue

        dispatch_tracker.record_dispatch(cand.get("track_ids", []), cand_t)

        cand_signals = cand.get("signals", {})
        cand_type = cand["event_type"]
        is_single = (cand_type in ("single_vehicle_anomaly", "single"))
        after_tel = getattr(ep, "aftermath_telemetry", {}) or {}
        acc_1 = float(cand_signals.get("accel_1_ms2", 0.0))
        acc_2 = float(cand_signals.get("accel_2_ms2", 0.0))
        min_acc = min(acc_1, acc_2) if (acc_1 != 0.0 or acc_2 != 0.0) else -4.0
        acc_val = float(cand_signals.get("metric_accel_ms2", min_acc))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        iou_val = float(cand_signals.get("bbox_iou", 0.10))
        norm_px = float(cand_signals.get("norm_pixel_dist", 0.35))
        is_calib = bool(cand_signals.get("is_calibrated", camera_cfg.is_calibrated))
        tau_rest = float(after_tel.get("tau_rest", cand_signals.get("tau_rest", cand_signals.get("stationary_duration_sec", 0.0))))
        spd_1 = float(cand_signals.get("speed_1_kmh", 0.0))
        spd_2 = float(cand_signals.get("speed_2_kmh", 0.0))
        v_entry = float(after_tel.get("v_entry", cand_signals.get("max_recent_speed_kmh", 30.0)))
        max_spd = max(spd_1, spd_2, v_entry)
        curr_spd = min(spd_1, spd_2) if (spd_1 > 0 or spd_2 > 0) else float(cand_signals.get("metric_speed_kmh", 0.0))
        speed_drop = float(after_tel.get("dv_sust", max(0.0, max_spd - curr_spd)))
        if speed_drop <= 0.0 and max_spd > 0.0:
            speed_drop = max_spd - curr_spd
        yaw_delta = float(after_tel.get("delta_theta", cand_signals.get("yaw_delta_deg", cand_signals.get("theta_deg", 30.0))))

        telemetry = {
            "video_id": video_id,
            "candidate_frame": cand_f,
            "event_type": cand_type,
            "involved_tracks": cand["track_ids"],
            "composite_score": float(cand_signals.get("composite_score", 0.85)),
            "accel_ms2": acc_val,
            "yaw_delta_deg": yaw_delta,
            "aspect_ratio_ratio": float(cand_signals.get("aspect_ratio_ratio", cand_signals.get("delta_ar", 0.20))),
            "d_edge_m": float(cand_signals.get("d_edge_m", 0.5)),
            "distance_m": dist_val,
            "norm_pixel_dist": norm_px,
            "bbox_iou": iou_val,
            "is_calibrated": is_calib,
            "homography_calibrated": is_calib,
            "anomaly_type": cand_signals.get("anomaly_type", cand_type),
            "max_recent_speed_kmh": max_spd,
            "speed_kmh": curr_spd,
            "speed_drop_kmh": speed_drop,
            "tau_rest": tau_rest,
            "stationary_duration_sec": tau_rest,
            "track_age": 15,
            "is_single_vehicle": is_single,
            "post_event_rest": cand_signals.get("post_event_rest", True),
            "shoulder_admitted": bool(cand_signals.get("shoulder_admitted", False)),
            "v_dot_n": float(cand_signals.get("v_dot_n", 0.0)),
            "S_dissip": float(cand_signals.get("S_dissip", 0.0)),
            "aftermath_score": float(getattr(ep, "aftermath_score", 0.0)),
        }

        res = arbitrator.arbitrate(
            evidence_strip=strip,
            telemetry=telemetry,
            candidate_frame=cand_f,
            video_id=video_id,
        )

        is_confirmed = (res.get("is_collision", False) or res.get("verdict") == "CONFIRMED_INCIDENT") and not res.get("suppressed", False)
        verdict_str = res.get("verdict", "UNKNOWN")
        if is_confirmed:
            status_tag = f"CONFIRMED ({res.get('source', '')})"
        elif verdict_str == "CANDIDATE_UNCONFIRMED":
            status_tag = f"UNCONFIRMED FALLBACK ({res.get('suppression_reason', '')})"
        else:
            status_tag = "REJECTED (FALSE ALARM)"

        print(f"  [{idx}/{len(episodes)}] Episode {ep.episode_id} @ F{cand_f:03d} -> {status_tag} (conf={res.get('confidence', 0.0):.2f}): {res.get('rationale', '')}")

        if is_confirmed:
            roi_filename = f"{video_id}_ep_{idx:02d}_f{cand_f:03d}_{clean_type}.jpg"
            roi_path = os.path.join(roi_dir, roi_filename)
            cv2.imwrite(roi_path, roi_card)

            sidecar_filename = f"{video_id}_ep_{idx:02d}_f{cand_f:03d}_{clean_type}.json"
            sidecar_path = os.path.join(roi_dir, sidecar_filename)
            with open(sidecar_path, "w", encoding="utf-8") as sf:
                json.dump({
                    "video_id": video_id,
                    "episode_id": ep.episode_id,
                    "candidate_frame": cand_f,
                    "timestamp_s": cand_t,
                    "incident_type": cand_type,
                    "confidence": float(res.get("confidence", conf_val)),
                    "track_ids": tids,
                    "timeline": {
                        "pre_frame": f_app,
                        "trigger_frame": f_imp,
                        "post_frame": f_post,
                        "dt_pre_sec": dt_pre_s,
                        "dt_post_sec": dt_post_s,
                    },
                    "telemetry": meta_dict,
                    "arbitration": res,
                    "fixed_spatial_roi": [rx1, ry1, rx2, ry2],
                    "actor_boxes_crop": {
                        "pre": actor_boxes_pre,
                        "trigger": actor_boxes_imp,
                        "post": actor_boxes_post,
                    },
                    "image_file": roi_filename,
                }, sf, indent=2)

            print(f"  [ROI TRIPTYCH] Saved confirmed 3-panel evidence card to: {roi_path}")

            arbitrated_incidents.append({
                "episode": ep,
                "candidate": cand,
                "arbitration": res,
            })

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Video:                   {resolved_path}")
    print(f"Frames Processed:        {last_f_idx}")
    print(f"Throughput:              {fps} FPS")
    print(f"Micro-Crops Evaluated:   {crops_evaluated}")
    print(f"Micro-Vehicles Found:    {micro_recovered}")
    print(f"Audited Detections:      {admitted_candidates}")
    print(f"Raw Proposals:           {len(all_candidates)}")
    print(f"Consolidated Episodes:   {len(episodes)}")
    print(f"Confirmed Incidents:     {len(arbitrated_incidents)}")
    for inc in arbitrated_incidents:
        c = inc["candidate"]
        a = inc["arbitration"]
        print(f"  * Frame {c['frame']} ({c['timestamp_s']}s) | {a.get('collision_type', c['event_type'])} | Conf: {a.get('confidence', 0.0):.2f}")
        print(f"    Rationale: {a.get('rationale', '')}")
    acc_mem = telemetry_acc.get_total_memory_usage()
    print(f"TelemetryAccumulator Peak Memory: {acc_mem['total_bytes']:,} bytes ({acc_mem['total_mib']} MiB) | Capacity: {acc_mem['allocated_capacity']:,} rows | Snapshots: {acc_mem['recorded_snapshots']:,} | Buffer: {acc_mem['buffer_bytes']:,} bytes | Reallocations: {acc_mem['reallocations']}")
    print("=" * 80)

    result_summary = {
        "video_id": video_id,
        "video_path": resolved_path,
        "frames_processed": last_f_idx,
        "fps": fps,
        "candidates_count": len(all_candidates),
        "confirmed_count": len(arbitrated_incidents),
        "crops_evaluated": crops_evaluated,
        "micro_recovered": micro_recovered,
        "admitted_recovered": admitted_candidates,
        "confirmed_incidents": arbitrated_incidents,
        "annotated_video_path": renderer.last_output_path if (renderer and save_video) else None,
        "telemetry_accumulator_memory": acc_mem,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)
        print(f"Saved results to: {output_json}")

    return result_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run Updated Road Surveillance Pipeline (Iteration N)")
    parser.add_argument("--video", "--video-path", "-v", type=str, required=True, help="Path to input video file or dataset video ID")
    parser.add_argument("--config", "--config-path", "-c", type=str, default="road_surveillance/config/default_config.yaml", help="Path to configuration yaml")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no-evidence", action="store_true", help="Do not save evidence filmstrip image")
    parser.add_argument("--save-video", action="store_true", default=True, help="Save annotated video to disk (default: True)")
    parser.add_argument("--no-save-video", action="store_false", dest="save_video", help="Disable saving annotated video to disk")
    parser.add_argument("--output-video", type=str, default=None, help="Custom output path for annotated video MP4 (default: output/videos/<video_id>_annotated.mp4)")
    parser.add_argument("--imgsz", "--resolution", type=str, default="native", help="Detector inference resolution (default: 'native' for unscaled feed, or integer e.g. 640)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_path=args.video,
        config_path=args.config,
        max_frames=args.max_frames,
        save_evidence=not args.no_evidence,
        save_video=args.save_video,
        output_video=args.output_video,
        output_json=args.output,
        imgsz=args.imgsz,
    )
