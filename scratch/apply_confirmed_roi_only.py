with open("run_pipeline.py", "r", encoding="utf-8") as f:
    code = f.read()

target_save_block = """        roi_filename = f"{video_id}_ep_{idx:02d}_f{cand_f:03d}_{clean_type}.jpg"
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
        strip = roi_card"""

repl_save_block = """        strip = roi_card"""

assert target_save_block in code, "target_save_block not found in run_pipeline.py"
code = code.replace(target_save_block, repl_save_block, 1)

target_confirmed_block = """        if is_confirmed:
            arbitrated_incidents.append({
                "episode": ep,
                "candidate": cand,
                "arbitration": res,
            })"""

repl_confirmed_block = """        if is_confirmed:
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
            })"""

assert target_confirmed_block in code, "target_confirmed_block not found in run_pipeline.py"
code = code.replace(target_confirmed_block, repl_confirmed_block, 1)

with open("run_pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully updated run_pipeline.py to save confirmed ROI crops only!")
