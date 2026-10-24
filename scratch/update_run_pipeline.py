with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''        boxes = []
        for f in [t_app, t_imp, t_post]:
            if f in frame_tracks_cache:
                boxes.extend(frame_tracks_cache[f].values())

        roi = compute_union_roi(boxes, (vid_h, vid_w), margin_factor=0.6, min_size=260)
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

        # Save single authoritative annotated incident ROI card + JSON metadata to output/roi_crops/
        cand_signals = cand.get("signals", {})
        cand_type = cand.get("event_type", "incident")
        clean_type = cand_type.lower().replace(" ", "_").replace("/", "_")
        conf_val = float(cand.get("confidence", 0.0))
        acc_val = float(cand_signals.get("metric_accel_ms2", cand_signals.get("accel_1_ms2", -4.0)))
        dist_val = float(cand_signals.get("ground_distance_m", cand_signals.get("distance_m", 1.0)))
        spd_val = float(cand_signals.get("metric_speed_kmh", cand_signals.get("speed_kmh", 0.0)))
        tids = cand.get("track_ids", [])

        # Build actor boxes relative to crop coordinates
        actor_boxes_crop = {}
        if t_imp in frame_tracks_cache:
            for tid, (bx1, by1, bx2, by2) in frame_tracks_cache[t_imp].items():
                if tid in tids:
                    cx1 = max(0, min(crop_impact.shape[1] - 1, int(round(bx1 - rx1))))
                    cy1 = max(0, min(crop_impact.shape[0] - 1, int(round(by1 - ry1))))
                    cx2 = max(0, min(crop_impact.shape[1] - 1, int(round(bx2 - rx1))))
                    cy2 = max(0, min(crop_impact.shape[0] - 1, int(round(by2 - ry1))))
                    if cx2 > cx1 and cy2 > cy1:
                        actor_boxes_crop[tid] = (cx1, cy1, cx2, cy2)'''

new_block = '''        # Extract candidate telemetry and involved actor track IDs
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
                    actor_boxes_crop[tid] = (cx1, cy1, cx2, cy2)'''

old_block_crlf = old_block.replace('\r\n', '\n').replace('\n', '\r\n')
old_block_lf = old_block.replace('\r\n', '\n')

if old_block_crlf in text:
    text = text.replace(old_block_crlf, new_block.replace('\n', '\r\n'))
    with open('run_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Updated run_pipeline.py CRLF')
elif old_block_lf in text:
    text = text.replace(old_block_lf, new_block)
    with open('run_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Updated run_pipeline.py LF')
else:
    print('Block not found')
