import cv2
import numpy as np
from road_surveillance.visualization import FrameAnnotator, PlaybackRenderer

annotator = FrameAnnotator()
img = np.zeros((720, 1280, 3), dtype=np.uint8)
annotated = annotator.annotate(
    image=img,
    tracklets=[],
    tracks={},
    kinematics={},
    active_candidates=[],
    fps=30.0,
    frame_idx=1,
    pts_ns=33333333,
)
print("Annotated shape:", annotated.shape)
renderer = PlaybackRenderer(display=False, save_processed_video=True, output_dir="output/videos")
renderer.setup_video_writer("test_clip", 30.0, 1280, 720)
renderer.render(annotated, 30.0)
renderer.close()
print("Renderer success!")
