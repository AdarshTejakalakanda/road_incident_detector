import re

with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add collections import if not present
if "import collections" not in content:
    content = content.replace("import time\n", "import time\nimport collections\n", 1)

# 2. Add extract_actor_crop_boxes, build_triptych_evidence_card, get_frame_image
target_func = '''def load_frame_at(video_path: str, target_frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_frame_idx - 1))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return np.zeros((720, 1280, 3), dtype=np.uint8)


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
    """Builds a single, self-contained incident evidence ROI card with HUD banner and bounding boxes."""
    canvas = crop_img.copy()
    h, w = canvas.shape[:2]

    # 1. Draw actor bounding boxes and labels
    palette = [(0, 0, 255), (0, 215, 255), (255, 128, 0), (0, 255, 128)]
    for i, tid in enumerate(track_ids):
        box = actor_boxes_crop.get(tid)
        if box:
            bx1, by1, bx2, by2 = box
            color = palette[i % len(palette)]
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
            label = f"ID:{tid} {event_type.upper()}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, by1 - th - 6)
            tag_y2 = by1
            tag_x1 = bx1
            tag_x2 = min(w, bx1 + tw + 6)
            cv2.rectangle(canvas, (tag_x1, tag_y1), (tag_x2, tag_y2), (20, 20, 20), -1)
            cv2.putText(canvas, label, (tag_x1 + 3, tag_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # 2. Build top metadata HUD banner
    banner_h = 58
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    cv2.rectangle(banner, (0, 0), (w, banner_h), (20, 20, 25), -1)
    cv2.rectangle(banner, (0, 0), (w, 3), (0, 165, 255), -1)

    title = f"[INCIDENT: {event_type.upper()}]  F{candidate_frame:03d} ({timestamp_s:.2f}s)  Clip: {video_id}"
    cv2.putText(banner, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    spd = float(telemetry.get("speed_kmh", 0.0))
    accel = float(telemetry.get("accel_ms2", 0.0))
    dist = float(telemetry.get("distance_m", 0.0))
    conf = float(telemetry.get("confidence", 0.0))
    metrics_str = f"Actors: {track_ids} | Spd: {spd:.1f} km/h | Accel: {accel:+.1f} m/s2 | Dist: {dist:.1f}m | Conf: {conf:.2f}"
    cv2.putText(banner, metrics_str, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([banner, canvas])'''

new_func = '''def load_frame_at(video_path: str, target_frame_idx: int) -> np.ndarray:
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
    )'''

assert target_func in content, "target_func not found"
content = content.replace(target_func, new_func, 1)

# 3. Add frame_ring_buffer in stream loop
target_loop_init = '''    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    all_candidates: List[Dict[str, Any]] = []
    crops_evaluated = 0
    micro_recovered = 0
    admitted_candidates = 0

    ingestor = DatasetVideoIngestor(path=resolved_path)
    t0 = time.perf_counter()
    last_f_idx = 0'''

new_loop_init = '''    frame_tracks_cache: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
    all_candidates: List[Dict[str, Any]] = []
    crops_evaluated = 0
    micro_recovered = 0
    admitted_candidates = 0

    # Rolling ring buffer for pre-event retrieval (e.g. 6.5 seconds @ stream_fps)
    ring_buf_len = max(60, int(round(stream_fps * 6.5)))
    frame_ring_buffer: collections.deque = collections.deque(maxlen=ring_buf_len)

    ingestor = DatasetVideoIngestor(path=resolved_path)
    t0 = time.perf_counter()
    last_f_idx = 0'''

assert target_loop_init in content, "target_loop_init not found"
content = content.replace(target_loop_init, new_loop_init, 1)

target_frame_store = '''        last_f_idx = f_idx
        img = frame.image'''

new_frame_store = '''        last_f_idx = f_idx
        img = frame.image
        frame_ring_buffer.append((f_idx, img.copy()))'''

assert target_frame_store in content, "target_frame_store not found"
content = content.replace(target_frame_store, new_frame_store, 1)

# 4. Update Aftermath Evaluation to use up to 5.0 seconds
target_aftermath = '''        if delta_t_avail < 0.75:
            res = SceneEscalationResult(
                disposition=SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW,
                score=None,
                window_state="INSUFFICIENT_AFTERMATH_WINDOW",
                telemetry={"delta_t_avail": delta_t_avail},
            )
        else:
            t_end = min(t_anchor + 1.50, t_stream_end)
            window_history = telemetry_acc.get_window(t_start=t_anchor, t_end=t_end)

            res = scene_escalator.evaluate_candidate(
                candidate=ep.peak_candidate,
                track_history=window_history,
                fps=stream_fps,
            )
            res.window_state = (
                "COMPLETE_AFTERMATH_WINDOW" if delta_t_avail >= 1.50
                else "PARTIAL_AFTERMATH_WINDOW"
            )'''

new_aftermath = '''        if delta_t_avail < 0.75:
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
            )'''

assert target_aftermath in content, "target_aftermath not found"
content = content.replace(target_aftermath, new_aftermath, 1)

# 5. Update Evidence Card Generation to 3-panel Fixed Spatial ROI Option B with 5s gap
target_card_gen = '''    for idx, ep in enumerate(episodes, 1):
        cand = ep.peak_candidate
        cand_f = ep.peak_frame
        cand_t = cand.get("timestamp_s", cand_f / 30.0)
        t_app = max(1, cand_f - 8)
        t_imp = cand_f
        t_post = min(last_f_idx, cand_f + 8)

        img_app = load_frame_at(resolved_path, t_app)
        img_imp = load_frame_at(resolved_path, t_imp)
        img_post = load_frame_at(resolved_path, t_post)

        # Extract candidate telemetry and involved actor track IDs
        cand_signals = cand.get("signals", {})
        cand_type = cand.get("event_type", "incident")
        clean_type = cand_type.lower().replace(" ", "_").replace("/", "_")
        conf_val = float(cand.get("confidence", 0.0))
        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -4.0)))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        spd_val = float(cand_signals.get("metric_speed_kmh", cand_signals.get("speed_kmh", 0.0)))
        tids = cand.get("track_ids", [])

        # Tight incident ROI crop centered strictly on involved actor track boxes
        boxes = []
        for f in [t_app, t_imp, t_post]:
            if f in frame_tracks_cache:
                for tid in tids:
                    if tid in frame_tracks_cache[f]:
                        boxes.append(frame_tracks_cache[f][tid])

        if not boxes:
            for f in range(max(1, t_imp - 5), min(last_f_idx + 1, t_imp + 6)):
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
            boxes = [(cx - 80, cy - 80, cx + 80, cy + 80)]

        roi = compute_union_roi(boxes, (vid_h, vid_w), margin_factor=0.5, min_size=260)
        rx1, ry1, rx2, ry2 = roi

        crop_pre = img_app[ry1:ry2, rx1:rx2]
        crop_impact = img_imp[ry1:ry2, rx1:rx2]
        crop_post = img_post[ry1:ry2, rx1:rx2]

        strip = create_evidence_strip(
            crop_pre=crop_pre,
            crop_impact=crop_impact,
            crop_post=crop_post,
            target_h=336,
            labels=(f"1: APPROACH (F{t_app})", f"2: INCIDENT (F{t_imp})", f"3: POST (F{t_post})"),
        )

        # Build actor boxes relative to crop coordinates
        actor_boxes_crop = {}
        for tid in tids:
            found_box = None
            for sf in [t_imp, t_app, t_post]:
                if sf in frame_tracks_cache and tid in frame_tracks_cache[sf]:
                    found_box = frame_tracks_cache[sf][tid]
                    break
            if found_box is None:
                for sf in range(max(1, t_imp - 5), min(last_f_idx + 1, t_imp + 6)):
                    if sf in frame_tracks_cache and tid in frame_tracks_cache[sf]:
                        found_box = frame_tracks_cache[sf][tid]
                        break
            if found_box is not None:
                bx1, by1, bx2, by2 = found_box
                cx1 = max(0, min(crop_impact.shape[1] - 1, int(round(bx1 - rx1))))
                cy1 = max(0, min(crop_impact.shape[0] - 1, int(round(by1 - ry1))))
                cx2 = max(0, min(crop_impact.shape[1] - 1, int(round(bx2 - rx1))))
                cy2 = max(0, min(crop_impact.shape[0] - 1, int(round(by2 - ry1))))
                if cx2 > cx1 and cy2 > cy1:
                    actor_boxes_crop[tid] = (cx1, cy1, cx2, cy2)

        meta_dict = {
            "speed_kmh": spd_val,
            "accel_ms2": acc_val,
            "distance_m": dist_val,
            "confidence": conf_val,
        }

        if crop_impact is not None and crop_impact.size > 0:
            roi_card = build_incident_roi_card(
                crop_img=crop_impact,
                video_id=video_id,
                candidate_frame=cand_f,
                timestamp_s=cand_t,
                event_type=cand_type,
                track_ids=tids,
                actor_boxes_crop=actor_boxes_crop,
                telemetry=meta_dict,
            )

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
                    "confidence": conf_val,
                    "track_ids": tids,
                    "telemetry": meta_dict,
                    "roi_box": [rx1, ry1, rx2, ry2],
                    "actor_boxes_in_roi": actor_boxes_crop,
                    "image_file": roi_filename,
                }, sf, indent=2)

            print(f"  [ROI CROP] Saved incident ROI card to: {roi_path}")'''

new_card_gen = '''    for idx, ep in enumerate(episodes, 1):
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
                "confidence": conf_val,
                "track_ids": tids,
                "timeline": {
                    "pre_frame": f_app,
                    "trigger_frame": f_imp,
                    "post_frame": f_post,
                    "dt_pre_sec": dt_pre_s,
                    "dt_post_sec": dt_post_s,
                },
                "telemetry": meta_dict,
                "fixed_spatial_roi": [rx1, ry1, rx2, ry2],
                "actor_boxes_crop": {
                    "pre": actor_boxes_pre,
                    "trigger": actor_boxes_imp,
                    "post": actor_boxes_post,
                },
                "image_file": roi_filename,
            }, sf, indent=2)

        print(f"  [ROI TRIPTYCH] Saved 3-panel evidence card to: {roi_path}")
        strip = roi_card'''

assert target_card_gen in content, "target_card_gen not found"
content = content.replace(target_card_gen, new_card_gen, 1)

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated run_pipeline.py successfully")
