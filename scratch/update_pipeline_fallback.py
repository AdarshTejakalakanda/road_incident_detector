with open('run_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''        if res.disposition in {
            SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT,
            SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW,
        }:
            escalated_episodes.append(ep)
            print(f"  [ESCALATED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> {res.disposition.value} (score={res.score}, type={res.canonical_type})")
        else:
            ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
            quarantined_records.append(suppression_entry)
            print(f"  [SUPPRESSED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (score={res.score}, type={res.canonical_type})")'''

repl = '''        if res.disposition == SceneEscalatorDisposition.ESCALATED_PHYSICAL_INCIDENT:
            escalated_episodes.append(ep)
            print(f"  [ESCALATED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> {res.disposition.value} (score={res.score}, type={res.canonical_type})")
        elif res.disposition == SceneEscalatorDisposition.FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW:
            cand_spd = float(ep.peak_candidate.get("signals", {}).get("metric_speed_kmh", ep.peak_candidate.get("speed_kmh", 0.0)))
            cand_conf = float(ep.peak_candidate.get("confidence", 0.0))
            if cand_spd > 8.0 and cand_conf >= 0.70:
                escalated_episodes.append(ep)
                print(f"  [ESCALATED FALLBACK] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> FALLBACK_INSUFFICIENT_AFTERMATH_WINDOW (spd={cand_spd:.1f} km/h)")
            else:
                ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
                quarantined_records.append(suppression_entry)
                print(f"  [SUPPRESSED FALLBACK] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (stationary/low-confidence boundary noise)")
        else:
            ep.state = "SUPPRESSED_BENIGN_AFTERMATH"
            quarantined_records.append(suppression_entry)
            print(f"  [SUPPRESSED] Episode {ep.episode_id} @ F{ep.peak_candidate['frame']:03d} -> SUPPRESSED_BENIGN_AFTERMATH (score={res.score}, type={res.canonical_type})")'''

assert target in content, "target not found"
content = content.replace(target, repl, 1)

with open('run_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated run_pipeline.py fallback handling successfully")
