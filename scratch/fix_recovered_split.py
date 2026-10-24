with open('road_surveillance/events/candidates/collision.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = """            # Multi-vehicle collision requires standard vehicle tracks or mature interaction-rescued tracks
            is_immature_cand = False
            for tr_chk in (tr1, tr2):
                if getattr(tr_chk, "is_tentative_rescued", False) and tr_chk.track_age < 3:
                    is_immature_cand = True
                    break
                if (getattr(tr_chk, "fused_class", "") == "candidate_vehicle_recovered" or tr_chk.class_label == "candidate_vehicle_recovered") and tr_chk.track_age < 2:
                    is_immature_cand = True
                    break
            if is_immature_cand:
                continue"""

replacement = """            # Multi-vehicle collision requires standard vehicle tracks or mature interaction-rescued tracks
            is_cand1 = (getattr(tr1, "fused_class", "") == "candidate_vehicle_recovered" or tr1.class_label == "candidate_vehicle_recovered")
            is_cand2 = (getattr(tr2, "fused_class", "") == "candidate_vehicle_recovered" or tr2.class_label == "candidate_vehicle_recovered")
            if is_cand1 and is_cand2:
                # Two recovered micro-crops cannot collide with each other
                self.persistence_counts[pair_key] = 0
                continue
            if (is_cand1 or is_cand2) and (iou >= 0.15 or dist <= 0.8 or norm_pixel_dist <= 0.40):
                # A recovered candidate overlapping an existing vehicle track without persistent closure
                if not (signal_decreasing_dist and (signal_convergence or severe_decel)):
                    self.persistence_counts[pair_key] = 0
                    self.log_suppressed(frame_idx, [t1_id, t2_id], "recovered_detector_split_duplicate", dist)
                    continue

            is_immature_cand = False
            for tr_chk in (tr1, tr2):
                if getattr(tr_chk, "is_tentative_rescued", False) and tr_chk.track_age < 3:
                    is_immature_cand = True
                    break
                if (getattr(tr_chk, "fused_class", "") == "candidate_vehicle_recovered" or tr_chk.class_label == "candidate_vehicle_recovered") and tr_chk.track_age < 5:
                    is_immature_cand = True
                    break
            if is_immature_cand:
                continue"""

text = text.replace(target, replacement)
with open('road_surveillance/events/candidates/collision.py', 'w', encoding='utf-8') as f:
    f.write(text)
print('Updated collision.py!')
