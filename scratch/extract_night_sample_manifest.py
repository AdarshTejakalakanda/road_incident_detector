import os
import sys
import hashlib
import json
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARCHIVE_ROOT = r"C:\Users\Adarsh Teja Kalakand\Downloads\archive"
CSV_PATH = os.path.join(ARCHIVE_ROOT, "metadata-real.csv")

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def main():
    df = pd.read_csv(CSV_PATH)
    
    # 3 distinct night-tagged clips
    selected_clips = [
        "real_videos/Z4kg2Ev3vhk_00.mp4",
        "real_videos/unS0-TLF1ao_00.mp4",
        "real_videos/UarP8qU1S-c_00.mp4",
    ]
    
    manifest = []
    
    for rel_path in selected_clips:
        row = df[df['path'] == rel_path].iloc[0]
        full_path = os.path.join(ARCHIVE_ROOT, rel_path)
        clip_id = os.path.basename(rel_path).replace(".mp4", "")
        
        cap = cv2.VideoCapture(full_path)
        actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        actual_duration = actual_frames / fps if fps > 0 else 0.0
        
        meta_frames = int(row['no_frames'])
        meta_duration = float(row['duration'])
        accident_frame = int(row['accident_frame'])
        
        # 5 frames spanning the accident window: [accident_frame - 20, accident_frame - 10, accident_frame, accident_frame + 10, accident_frame + 20]
        # clamped to [0, actual_frames - 1]
        offsets = [-20, -10, 0, 10, 20]
        window_frames = [max(0, min(actual_frames - 1, accident_frame + o)) for o in offsets]
        
        extracted_frames_info = []
        for f_idx in window_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            assert ret and frame is not None, f"Failed to extract frame {f_idx} from {rel_path}"
            
            # Compute luminance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lum = float(np.mean(gray))
            
            # Encode to raw bytes for SHA-256
            frame_bytes = frame.tobytes()
            f_hash = sha256_bytes(frame_bytes)
            
            extracted_frames_info.append({
                "frame_index": f_idx,
                "offset_from_accident": f_idx - accident_frame,
                "shape": list(frame.shape),
                "mean_luminance": round(lum, 3),
                "sha256": f_hash,
            })
            
        cap.release()
        
        duration_match = abs(actual_duration - meta_duration) < 0.1
        frames_match = (actual_frames == meta_frames or abs(actual_frames - meta_frames) <= 1)
        
        manifest.append({
            "clip_id": clip_id,
            "path": rel_path,
            "region": str(row['region']),
            "scene_layout": str(row['scene_layout']),
            "weather": str(row['weather']),
            "accident_frame": accident_frame,
            "metadata_frames": meta_frames,
            "actual_frames": actual_frames,
            "metadata_duration_s": meta_duration,
            "actual_duration_s": round(actual_duration, 3),
            "duration_match": duration_match,
            "frames_match": frames_match,
            "frames": extracted_frames_info,
        })
        
    out_path = "scratch/night_sample_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {out_path}")

if __name__ == "__main__":
    main()
