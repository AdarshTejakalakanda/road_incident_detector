"""Evaluation Harness for Nocturnal Light-Core Branch + ByteTrack Integration.

Evaluates vehicle-level recall, track stability, and ID consistency on night CCTV footage.
Saves annotated MP4, GIF, and performance telemetry to output/annotated_videos/.
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from road_surveillance.detection.nocturnal_light_branch import NocturnalLightBranchDetector
from road_surveillance.tracking.tracker import ByteTrackTracker
from tools.run_annotated_detector import resolve_video_path


UPPER_HIGHWAY_POLY = np.array([
    [320, 560],  # Bottom left
    [240, 480],  # Left edge
    [420, 315],  # Horizon left
    [700, 315],  # Horizon right
    [620, 400],  # Road edge before grass
    [540, 470],  # Road edge hugging pavement
    [480, 560],  # Bottom right pavement edge
], np.int32)


def evaluate_video(
    video_input: str = "yzNEliTlFGk_00",
    output_dir: str = r"output\annotated_videos",
):
    video_path = resolve_video_path(video_input)
    clip_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    out_mp4 = os.path.join(output_dir, f"LightCore_ByteTrack_{clip_id}.mp4")
    out_gif = os.path.join(output_dir, f"LightCore_ByteTrack_{clip_id}.gif")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.033
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 240

    detector = NocturnalLightBranchDetector(road_polygon=UPPER_HIGHWAY_POLY)
    tracker = ByteTrackTracker(track_activation_threshold=0.25, frame_rate=int(fps))

    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + 40))
    gif_frames = []

    print("================================================================")
    print("NOCTURNAL LIGHT-CORE BRANCH + BYTETRACK INTEGRATION EVALUATION")
    print(f"Target Video: {clip_id}")
    print(f"Resolution: {w}x{h} @ {fps:.2f} FPS ({total_frames} frames)")
    print(f"Output MP4: {out_mp4}")
    print(f"Output GIF: {out_gif}")
    print("================================================================\n")

    f_idx = 0
    total_raw_hypotheses = 0
    unique_track_ids = set()
    track_lifespans = {}
    incident_track_ids = set()

    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts_s = f_idx / fps
        batch = detector.detect_frame(frame, frame_idx=f_idx, timestamp_s=ts_s, video_id=clip_id)
        total_raw_hypotheses += len(batch.boxes)

        # Update ByteTrack tracker
        states, tracks = tracker.update(batch)

        ann = frame.copy()
        cv2.polylines(ann, [UPPER_HIGHWAY_POLY], isClosed=True, color=(80, 80, 80), thickness=1)

        active_tracks_count = 0
        for tid, trk in tracks.items():
            unique_track_ids.add(tid)
            track_lifespans[tid] = track_lifespans.get(tid, 0) + 1

            if trk.status == "active" and trk.lost_frame_count == 0:
                active_tracks_count += 1
                x, y, bw, bh = trk.current_bbox_xywh
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                
                # Check if this track is at the accident scene (y in [380, 520], x in [420, 600])
                if 380 <= y <= 520 and 400 <= x <= 620:
                    incident_track_ids.add(tid)
                    color = (0, 240, 120)  # Vivid Cyan/Green
                else:
                    color = (255, 180, 0)  # Blue/Gold

                cv2.rectangle(ann, (x1, y1), (x2, y2), color, 2)
                lbl = f"Track #{tid} [{trk.class_label}] {trk.detection_confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(ann, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (15, 23, 42), -1)
                cv2.putText(ann, lbl, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # HUD Bar
        hud = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.rectangle(hud, (0, 0), (w, 40), (15, 23, 42), -1)
        cv2.line(hud, (0, 39), (w, 39), (51, 65, 85), 1)

        hud_left = f"Light-Core Branch + ByteTrack Fusion | {clip_id}"
        hud_right = f"Frame: {f_idx:03d}/{total_frames} ({ts_s:.2f}s) | Active Tracks: {active_tracks_count}"
        cv2.putText(hud, hud_left, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(hud, hud_right, (w - 380, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (241, 245, 249), 1, cv2.LINE_AA)

        combined = np.vstack([hud, ann])
        writer.write(combined)

        if f_idx % 2 == 0:
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            gh, gw = rgb.shape[:2]
            gif_frames.append(Image.fromarray(cv2.resize(rgb, (gw // 2, gh // 2), interpolation=cv2.INTER_AREA)))

        f_idx += 1
        if f_idx % 60 == 0:
            print(f"  Processed {f_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    t_elapsed = time.perf_counter() - t0

    fps_achieved = total_frames / t_elapsed
    print(f"\nProcessing Completed in {t_elapsed:.2f}s ({fps_achieved:.1f} FPS)")
    print(f"Total Raw Vehicle Hypotheses: {total_raw_hypotheses} (Avg {total_raw_hypotheses/total_frames:.2f}/frame)")
    print(f"Total Unique Tracks Established: {len(unique_track_ids)}")
    print(f"Incident Scene Tracks (Collision Site): {sorted(list(incident_track_ids))}")

    # Long-lived persistent tracks
    stable_tracks = {tid: count for tid, count in track_lifespans.items() if count >= 10}
    print(f"Stable Long-Lived Tracks (>= 10 frames): {len(stable_tracks)}")
    for tid, count in sorted(stable_tracks.items()):
        print(f"  Track #{tid}: persisted {count} frames ({count/fps:.1f}s)")

    print("\nSaving Browser-Playable GIF...")
    if gif_frames:
        gif_frames[0].save(
            out_gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(2000 / fps),
            loop=0,
        )
    print("Done! Files saved to output/annotated_videos/")
    return out_mp4, out_gif


if __name__ == "__main__":
    evaluate_video()
