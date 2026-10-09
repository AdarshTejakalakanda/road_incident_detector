import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import cv2
import logging
import time
from typing import Optional

from road_surveillance.config import load_config, AppConfig
from road_surveillance.video import DatasetVideoIngestor, RollingFrameBuffer, IngestedFrame, probe_video_metadata
from road_surveillance.preprocessing import LowLightPreprocessor
from road_surveillance.detection import YOLO26Detector
from road_surveillance.tracking import RobustSurveillanceTracker
from road_surveillance.calibration import PlanarHomography, LaneManager, CameraConfiguration
from road_surveillance.calibration.road_surface import RoadSurfaceDetector
from road_surveillance.geometry import GeometryEngine
from road_surveillance.events import EventEngine
from road_surveillance.evidence import EvidenceStore, EpisodeEvidenceConsolidator
from road_surveillance.visualization import FrameAnnotator, PlaybackRenderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pipeline")


def run_pipeline(
    config_path: str = "road_surveillance/config/default_config.yaml",
    video_override: Optional[str] = None,
    display: Optional[bool] = None,
    save_video: Optional[bool] = None,
    max_videos: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    """Execute Phase 1 Road Surveillance Pipeline with Adaptive Nocturnal Sensitivity."""
    cfg: AppConfig = load_config(config_path)

    dataset_path = video_override if video_override else cfg.dataset.path
    show_display = display if display is not None else cfg.playback.display
    save_proc_video = save_video if save_video is not None else cfg.playback.save_processed_video

    logger.info("=" * 60)
    logger.info("ROAD SURVEILLANCE INCIDENT DETECTION SYSTEM - PHASE 1")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Detector checkpoint: {cfg.detection.model_path}")
    logger.info(f"Device: {cfg.detection.device}")

    # 1. Initialize Video Ingestor
    ingestor = DatasetVideoIngestor(
        path=dataset_path,
        recursive=cfg.dataset.recursive,
        extensions=cfg.dataset.extensions,
    )

    if not ingestor.video_files:
        logger.error(f"No video files found in {dataset_path} matching {cfg.dataset.extensions}")
        return

    logger.info(f"Discovered {len(ingestor.video_files)} video file(s) for processing.")

    # 2. Initialize YOLO26 Detector with Nocturnal Sensitivity
    detector = YOLO26Detector(
        model_path=cfg.detection.model_path,
        conf_threshold=cfg.detection.confidence_threshold,
        nocturnal_conf_threshold=0.10,
        enable_nocturnal_headlight_remap=True,
        input_resolution=cfg.detection.input_resolution,
        device=cfg.detection.device,
        frame_sampling_rate=cfg.detection.frame_sampling_rate,
    )

    # 3. Initialize Robust Surveillance Tracker
    tracker = RobustSurveillanceTracker(
        track_activation_threshold=0.10,
        low_confidence_floor=0.06,
        lost_track_buffer=cfg.tracking.lost_track_buffer,
        match_threshold_high=0.75,
        match_threshold_low=0.65,
    )

    # 4. Initialize Evidence Store & Collector
    evidence_store = EvidenceStore(root_dir=cfg.evidence.output_dir)

    # 5. Initialize Playback Renderer
    renderer = PlaybackRenderer(
        display=show_display,
        save_processed_video=save_proc_video,
        realtime=cfg.playback.realtime,
        output_dir=cfg.playback.output_dir,
    )

    video_count = 0
    total_candidates_found = 0

    try:
        for video_path in ingestor.video_files:
            video_count += 1
            if max_videos and video_count > max_videos:
                break

            camera_id, video_id = ingestor.generate_ids(video_path)
            logger.info(f"\n--- Processing Video [{video_count}/{len(ingestor.video_files)}]: {video_id} ({video_path}) ---")

            # 1. Probe video metadata (resolution, fps, duration)
            metadata = probe_video_metadata(video_path, camera_id=camera_id, video_id=video_id)
            w, h = metadata.resolution
            fps = metadata.fps

            # 2. Camera Configuration & Dynamic Homography
            is_calibrated = bool(cfg.calibration.get(video_id) or cfg.calibration.get(camera_id))
            raw_calib_cfg = cfg.calibration.get(video_id) or cfg.calibration.get(camera_id) or cfg.calibration.get("default")
            camera_cfg = CameraConfiguration.from_config_dict(
                camera_id=camera_id,
                cfg_obj=raw_calib_cfg,
                is_calibrated=is_calibrated,
            )

            # Build dynamic homography adapted to target video resolution without hardcoding
            homography = PlanarHomography.from_camera_config(camera_cfg, target_resolution=(w, h))

            # 3. Calibration validity check & horizon line estimation
            calib_val = homography.validate_calibration(image_resolution=(w, h))
            horizon_y = calib_val["horizon"].get("v_horizon_center")
            horizon_str = f"{horizon_y:.1f}px" if isinstance(horizon_y, (int, float)) else "N/A"

            logger.info(
                f"Camera setup for {video_id}: mode={camera_cfg.mode.value}, "
                f"resolution={w}x{h}@{fps:.1f}fps, homography_valid={calib_val['is_valid']}, "
                f"horizon_center={horizon_str}"
            )

            lane_manager = LaneManager.from_config(camera_cfg.lanes, homography)

            geometry_engine = GeometryEngine(
                homography=homography,
                lane_manager=lane_manager,
                smoothing_window=cfg.geometry.smoothing_window,
                stationary_speed_threshold_kmh=cfg.geometry.stationary_speed_threshold_kmh,
                is_camera_calibrated=camera_cfg.is_calibrated,
            )

            event_engine = EventEngine(
                config=cfg.events,
                available_vocabulary=detector.names,
                is_camera_calibrated=camera_cfg.is_calibrated,
            )
            event_engine.set_resolution(w, h)

            frame_buffer = RollingFrameBuffer(
                max_seconds=cfg.evidence.max_buffer_seconds,
            )

            consolidator = EpisodeEvidenceConsolidator(
                evidence_store=evidence_store,
                frame_buffer=frame_buffer,
                max_gap_frames=25,
                post_cooldown_frames=30,
                pre_event_seconds=cfg.evidence.pre_event_seconds,
                post_event_seconds=cfg.evidence.post_event_seconds,
            )

            annotator = FrameAnnotator(homography=homography, lane_manager=lane_manager)
            preprocessor = LowLightPreprocessor(mode="N3", low_light_threshold=50.0, gamma=0.65)
            road_surface = RoadSurfaceDetector()

            tracker.reset()
            first_frame = True
            video_candidates = []
            frame_counter = 0
            start_time = time.time()

            for frame in ingestor.stream_video(video_path):
                frame_counter += 1
                if max_frames and frame_counter > max_frames:
                    break

                if first_frame:
                    fh, fw = frame.image.shape[:2]
                    renderer.setup_video_writer(video_id=video_id, fps=frame.fps, width=fw, height=fh)
                    first_frame = False

                frame_buffer.append(frame)
                processed_img, illum_telem = preprocessor.process(frame.image)
                is_low_light = bool(illum_telem.get("mean_luminance", 100.0) < 50.0)

                detection_frame = IngestedFrame(
                    camera_id=frame.camera_id,
                    video_id=frame.video_id,
                    frame_idx=frame.frame_idx,
                    pts_ns=frame.pts_ns,
                    fps=frame.fps,
                    image=processed_img,
                )
                batch = detector.detect(detection_frame, is_low_light=is_low_light, road_surface=road_surface)
                if len(batch.boxes) > 0:
                    road_surface.register_vehicle_observations(
                        [(b[0], b[1], b[2] - b[0], b[3] - b[1]) for b in batch.boxes],
                        frame_shape=processed_img.shape[:2],
                    )

                tracklets, active_tracks = tracker.update(batch)
                kinematics, convergences = geometry_engine.process_frame(tracklets, active_tracks)

                candidates = event_engine.process_frame(
                    frame_idx=frame.frame_idx,
                    pts_ns=frame.pts_ns,
                    video_id=video_id,
                    tracks=active_tracks,
                    kinematics=kinematics,
                    convergences=convergences,
                    lane_manager=lane_manager,
                )

                annotated = annotator.annotate(
                    image=frame.image,
                    tracklets=tracklets,
                    tracks=active_tracks,
                    kinematics=kinematics,
                    active_candidates=candidates,
                    fps=frame.fps,
                    frame_idx=frame.frame_idx,
                    pts_ns=frame.pts_ns,
                )

                current_telemetry = {
                    "frame_idx": frame.frame_idx,
                    "pts_ns": frame.pts_ns,
                    "tracks": {
                        tid: {
                            "label": tr.class_label,
                            "bbox_xywh": tr.current_bbox_xywh,
                            "ground_pos": kin.ground_pos,
                            "speed_kmh": kin.speed_kmh,
                            "accel_mag": kin.acceleration_mag,
                            "heading_deg": kin.heading_deg,
                            "stationary_sec": kin.stationary_duration_sec,
                            "lane_id": kin.lane_id,
                        }
                        for tid, kin in kinematics.items()
                        if (tr := active_tracks.get(tid)) is not None
                    },
                }

                for cand in candidates:
                    video_candidates.append(cand)
                    consolidator.on_event_candidate(
                        candidate=cand,
                        current_frame=frame,
                        annotated_image=annotated,
                        current_telemetry=current_telemetry,
                    )

                consolidator.process_frame(frame, current_telemetry)

                should_continue = renderer.render(annotated, frame.fps)
                if not should_continue:
                    logger.info("Processing halted by user.")
                    return

            consolidator.flush_all()
            completed_count = len(consolidator.completed_episodes)
            elapsed = time.time() - start_time
            proc_fps = frame_counter / elapsed if elapsed > 0 else 0.0

            logger.info(
                f"Completed video {video_id}: {frame_counter} frames in {elapsed:.1f}s "
                f"({proc_fps:.1f} FPS). Raw triggers: {len(video_candidates)}, "
                f"Consolidated episodes: {completed_count}"
            )
            total_candidates_found += completed_count

    finally:
        renderer.close()
        logger.info("\n" + "=" * 60)
        logger.info(f"Pipeline completed: Processed {video_count} video(s). Total candidates: {total_candidates_found}")
        logger.info(f"Evidence stored in: {evidence_store.root_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Road Surveillance Incident Detection System")
    parser.add_argument("--config", default="road_surveillance/config/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--video", default=None, help="Path to single incident video or directory")
    parser.add_argument("--display", action="store_true", help="Display live annotated playback window")
    parser.add_argument("--no-save-video", action="store_true", help="Disable saving processed video to disk")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per video to process")

    args = parser.parse_args()
    save_video_flag = False if args.no_save_video else True

    run_pipeline(
        config_path=args.config,
        video_override=args.video,
        display=args.display if args.display else None,
        save_video=save_video_flag,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
    )
