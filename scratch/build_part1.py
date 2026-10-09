import base64,os,sys

block1 = """\"\"\"VLM Arbitration Module with Composite Evidence Filmstrip, Strict Stage T Telemetry Gating, and Groq Integration.

Iteration K.5 / Stage T: Video-VLM Arbitration Layer
- Creates a single-composite horizontal triplet filmstrip (Approach, Impact, Post)
  to stay within Groq's single-image token and rate budgets.
- Stage T Corrected Tier 1 Admission: Quarantined candidates must pass strict physical gating
  (KinematicDissipation AND SpatialConvergence) before dispatch to VLM.
- Asymmetric VLM Arbitration: High bar to confirm, definitive bar to reject, quarantine otherwise.
- Preserves deterministic high-confidence bypass (VLM cannot override).
\"\"\"

import os
import time
import json
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any
from modules.evidence_builder import (
    select_temporal_evidence_frames,
    render_actor_overlays,
    create_stage_t_composite_strip,
)
import cv2
import numpy as np

logger = logging.getLogger("modules.vlm_arbitrator")

def create_evidence_strip(
    crop_pre: np.ndarray,
    crop_impact: np.ndarray,
    crop_post: np.ndarray,
    target_h: int = 336,
    labels: Tuple[str, str, str] = ("1: APPROACH", "2: CANDIDATE IMOACT", "3: POST-EVENT"),
) -> np.ndarray:
    panels = []
    crops = [crop_pre, crop_impact, crop_post]

    for i, crop in enumerate(crops):
        if crop is None or crop.size == 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)

        h, w = crop.shape[:2]
        if h <= 0 or"w <= 0:
            crop = np.zeros((target_h, target_h, 3), dtype=np.uint8)
            h, w = target_h, target_h

        new_w = max(10, int(w * (target_h / float(h))))
        resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_AREA)

        banner = np.zeros((28, new_w, 3), dtype=np.uint8)
        label = labels[i] if i < len(labels) else f"STAGE {i+1}"
        cv2.putText(banner, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        panel = np.vstack([banner, resized])
        panels.append(panel)

    divider = np.full((target_h + 28, 3, 3), 180, dtype=np.uint8)
    strip_elements = []
    for i, p in enumerate(panels):
        strip_elements.append(p)
        if i < len(panels) - 1:
            strip_elements.append(divider)

    return np.hstack(strip_elements)

"""
with open('modules/vlm_arbitrator.py', 'w', encoding='utf-8') as f:
    f.write(block1)
print('Part 1 written')
