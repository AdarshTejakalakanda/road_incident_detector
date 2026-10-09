import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from modules.evidence_builder import select_temporal_evidence_frames, render_actor_overlays, create_stage_t_composite_strip
from road_surveillance.video.ingestion import DatasetVideoIngestor

TARGET_CASES = [
    {
        "episode_id": "ftk_fSc4haM_00_ep01_f127",
        "video_id": "ftk_fSc4haM_00",
        "frame_index": 127,
        "incident_class": "Nocturnal Crossing T-Bone",
        "weather": "normal",
        "day_time": "night",
    },
    {
        "episode_id": "BgdKGnFqz0I_00_ep01_f126",
        "video_id": "BgdKGnFqz0I_00",
        "frame_index": 126,
        "incident_class": "Daylight Intersection T-Bone",
        "weather": "normal",
        "day_time": "day",
    },
    {
        "episode_id": "iA_u73BG4ZM_00_ep01_f133",
        "video_id": "iA_u73BG4ZM_00",
        "frame_index": 133,
        "incident_class": "Parallel Rear-End Collision",
        "weather": "normal",
        "day_time": "day",
    },
    {
        "episode_id": "UarP8qU1S-c_01_ep01_f63",
        "video_id": "UarP8qU1S-c_01",
        "frame_index": 63,
        "incident_class": "Nocturnal Guardrail / Multi-Vehicle Contact",
        "weather": "normal",
        "day_time": "night",
    },
    {
        "episode_id": "vxZ47DoPNY8_00_ep01_f55",
        "video_id": "vxZ47DoPNY8_00",
        "frame_index": 55,
        "incident_class": "Adverse Weather (Rain) Multi-Vehicle Impact",
        "weather": "rain",
        "day_time": "day",
    },
]

def build_filmstrips():
    video_dir = r"C:\Users\KALAKA~1\Downloads\archive\real_videos"
    os.makedirs("artifacts/stage_t_strips", exist_ok=True)
    strips_info = []

    for item in TARGET_CASES:
        vid = item["video_id"]
        cand_f = item["frame_index"]
        vpath = os.path.join(video_dir, f"{vid}.mp4")

        ingestor = DatasetVideoIngestor(path=vpath)
        frame_pts_map = {}
        frames_dict = {}

        # Read frames up to cand_f + 45
        for frame in ingestor.stream_video(vpath):
            idx = frame.frame_idx
            pts_s = float(frame.pts_ns * 1e-9)
            frame_pts_map[idx] = pts_s
            if abs(idx - cand_f) <= 45:
                frames_dict[idx] = frame.image.copy()
            if idx > cand_f + 45:
                break

        cap = cv2.VideoCapture(vpath)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()
        cand_t = frame_pts_map.get(cand_f, cand_f / fps)
        f_pre, f_imp, f_post = select_temporal_evidence_frames(
            candidate_frame=cand_f,
            candidate_timestamp_s=cand_t,
            fps=fps,
            total_frames=len(frame_pts_map),
            frame_pts_map=frame_pts_map,
            delta_pre_s=1.0,
            delta_post_s=1.5,
        )

        img_pre = frames_dict.get(f_pre)
        img_imp = frames_dict.get(f_imp)
        img_post = frames_dict.get(f_post)

        print(f"[{vid}] cand_f={cand_f} (t={cand_t:.2f}s) -> f_pre={f_pre} (t={frame_pts_map.get(f_pre,0):.2f}s), f_imp={f_imp}, f_post={f_post} (t={frame_pts_map.get(f_post,0):.2f}s)")

        strip = create_stage_t_composite_strip(
            img_pre=img_pre,
            img_impact=img_imp,
            img_post=img_post,
            target_h=336,
            labels=(f"1: PRE-EVENT (F{f_pre}, t-1.0s)", f"2: CANDIDATE IMPACT (F{f_imp}, t)", f"3: POST-EVENT (F{f_post}, t+1.5s)"),
        )

        strip_path = f"artifacts/stage_t_strips/{vid}_f{cand_f}_strip.jpg"
        cv2.imwrite(strip_path, strip, [cv2.IMWRITE_JPEG_QUALITY, 90])

        strips_info.append({
            "target": item,
            "strip_path": strip_path,
            "f_pre": f_pre,
            "f_imp": f_imp,
            "f_post": f_post,
            "t_pre": frame_pts_map.get(f_pre),
            "t_imp": cand_t,
            "t_post": frame_pts_map.get(f_post),
            "fps": fps,
        })

    with open("scratch/stage_t_strips_info.json", "w", encoding="utf-8") as f:
        json.dump(strips_info, f, indent=2)
    print("All 5 filmstrips constructed successfully.")

if __name__ == "__main__":
    build_filmstrips()
