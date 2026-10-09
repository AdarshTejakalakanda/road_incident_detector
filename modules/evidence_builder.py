import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

def select_temporal_evidence_frames(
    candidate_frame: int,
    candidate_timestamp_s: float,
    fps: float,
    total_frames: Optional[int] = None,
    frame_pts_map: Optional[Dict[int, float]] = None,
    delta_pre_s: float = 1.0,
    delta_post_s: float = 1.5,
) -> Tuple[int, int, int]:
    f_impact = int(candidate_frame)
    if frame_pts_map and len(frame_pts_map) > 0:
        target_pre_s = candidate_timestamp_s - delta_pre_s
        target_post_s = candidate_timestamp_s + delta_post_s
        pre_candidates = [f for f in frame_pts_map.keys() if f < f_impact]
        if pre_candidates:
            f_pre = min(pre_candidates, key=lambda f: abs(frame_pts_map[f] - target_pre_s))
        else:
            f_pre = max(1, f_impact - 1)
        post_candidates = [f for f in frame_pts_map.keys() if f > f_impact]
        if post_candidates:
            f_post = min(post_candidates, key=lambda f: abs(frame_pts_map[f] - target_post_s))
        else:
            f_post = f_impact + 1
    else:
        effective_fps = max(1.0, float(fps))
        offset_pre = max(1, int(round(delta_pre_s * effective_fps)))
        offset_post = max(1, int(round(delta_post_s * effective_fps)))
        f_pre = max(1, f_impact - offset_pre)
        f_post = f_impact + offset_post
        if total_frames is not None and total_frames > 0:
            f_post = min(total_frames, f_post)
    return f_pre, f_impact, f_post

def render_actor_overlays(
    image: np.ndarray,
    actor_boxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    stroke: int = 2,
    color_palette: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    annotated = image.copy()
    if not actor_boxes:
        return annotated
    palette = color_palette or [
        (0, 255, 255),
        (255, 128, 0),
        (0, 255, 0),
        (255, 0, 255),
        (0, 165, 255),
    ]
    h_img, w_img = annotated.shape[:2]
    for idx, (actor_id, box) in enumerate(actor_boxes.items()):
        if box is None or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = palette[idx % len(palette)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, stroke)
        label = f'ID:{actor_id}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        tag_y1 = max(0, y1 - txt_h - 4)
        tag_y2 = y1
        tag_x1 = x1
        tag_x2 = min(w_img, x1 + txt_w + 4)
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), (20, 20, 20), -1)
        cv2.putText(annotated, label, (tag_x1 + 2, tag_y2 - 2), font, font_scale, color, font_thickness, cv2.LINE_AA)
    return annotated

def create_stage_t_composite_strip(
    img_pre: np.ndarray,
    img_impact: np.ndarray,
    img_post: np.ndarray,
    target_h: int = 336,
    labels: Tuple[str, str, str] = ('1: PRE-EVENT (t - 1.0s)', '2: CANDIDATE IMPACT (t)', '3: POST-EVENT (t + 1.5s)'),
) -> np.ndarray:
    panels = []
    crops = [img_pre, img_impact, img_post]
    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            h, w = target_h, target_h
        aspect = float(w) / float(h)
        new_w = max(40, int(round(target_h * aspect)))
        resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)
        banner_h = 30
        cv2.rectangle(resized, (0, 0), (new_w, banner_h), (25, 25, 25), -1)
        lbl = labels[i] if i < len(labels) else f'STAGE {i+1}'
        text_color = (0, 255, 255) if i == 1 else (255, 255, 255)
        cv2.putText(resized, lbl, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        cv2.rectangle(resized, (0, 0), (new_w - 1, target_h - 1), (60, 60, 60), 1)
        panels.append(resized)
    divider = np.full((target_h, 3, 3), 40, dtype=np.uint8)
    composite = np.hstack([panels[0], divider, panels[1], divider, panels[2]])
    return composite
