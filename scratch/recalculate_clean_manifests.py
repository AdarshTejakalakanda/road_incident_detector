import os
import sys
import json
import hashlib
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

QUARANTINE_CLIPS = {
    "5uh8-3q51uI_00": "Used in night diagnostic and day baseline investigation",
    "82sQ7N-aHOw_00": "Used in test_82sQ7N_run.py and Phase 1 report forensic failure tables",
    "BgdKGnFqz0I_00": "Used in benchmark sweeps and track forensics",
    "F0NHENn-HS4_00": "Used in prototype collision tuning scripts",
    "KTGByI3N4G0_00": "Used in diagnose_ktg.py, run_ktg_test.py, diagnose_ktg_details.py",
    "LiM5LPSZoow_00": "Used in benchmark regression ablation scripts",
    "OXVXsdDT9QQ_00": "Used in run_stage_t_cohort_validation.py (QUAR-18)",
    "UarP8qU1S-c_00": "Used in trace_uarp.py, debug sweeps, and calibration verification",
    "UarP8qU1S-c_01": "Used in trace_uarp.py and multi-sweep benchmark scripts",
    "VozrseCzhQc_00": "Used in solo crash failure taxonomy and regression tests",
    "Z4kg2Ev3vhk_00": "Used in debug_z4_f170.py, check_z4_episodes.py, night diagnostic",
    "aF2Hbp8C23A_00": "Used in prototype single-vehicle tuning",
    "aPsS_HaokoU_00": "Used in motion ROI gate tuning and forensics",
    "acmBtouCwIU_00": "Used in collision candidate ablation scripts",
    "bJMc3sNP2AU_00": "Used in nocturnal recovery and solo vehicle tracking benchmarks",
    "ftk_fSc4haM_00": "Used in nocturnal regression benchmark scripts",
    "fzWY0vLAXzI_00": "Used in kinematic maturity ablation scripts",
    "iA_u73BG4ZM_00": "Used in inspect_raw_ia_dets.py, trace_ia_collision.py",
    "kABWTDKgc9U_00": "Used in run_stage_t_cohort_validation.py single vehicle anomaly tests",
    "maJGw3hQs_s_00": "Used in tracklet fragmentation analysis scripts",
    "maJsgJsapH4_00": "Used in run_stage_t_cohort_validation.py intersection turn tests",
    "nAXTthLfgtI_00": "Used in diagnose_naxt.py, nocturnal regression sweeps",
    "unS0-TLF1ao_00": "Used in intersection diagnostic sweeps and night sample manifest",
    "vxZ47DoPNY8_00": "Used in crossing traffic conflict ablation",
    "yzLkvjtWFeI_00": "Used in benchmark L and M sweeps",
    "yzNEliTlFGk_00": "Used in nocturnal candidate generation scripts",
    "zOgTWOFJpKg_00": "Used in nocturnal threshold ladder validation",
}

def main():
    with open("scratch/cohort_manifests.json") as f:
        data = json.load(f)
        
    old_cohorts = data["cohorts"]
    clean_cohorts = {
        "verified-night": [],
        "mislabeled-night": [],
        "day": [],
        "excluded": [],
        "quarantined-tuning": [],
    }
    
    for cname, items in old_cohorts.items():
        for item in items:
            cid = item["clip_id"]
            if cid in QUARANTINE_CLIPS:
                item_copy = dict(item)
                item_copy["pre_quarantine_cohort"] = cname
                item_copy["quarantine_reason"] = QUARANTINE_CLIPS[cid]
                clean_cohorts["quarantined-tuning"].append(item_copy)
            else:
                clean_cohorts[cname].append(item)
                
    counts = {k: len(v) for k, v in clean_cohorts.items()}
    total = sum(counts.values())
    assert total == 2027, f"Total mismatch: {total} != 2027"
    
    payload = {
        "remediation_status": "QUARANTINE_APPLIED",
        "total_clips": total,
        "clean_counts": counts,
        "cohorts": clean_cohorts
    }
    
    payload_bytes = json.dumps(payload, indent=2).encode('utf-8')
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    payload["clean_manifest_sha256"] = payload_hash
    
    out_path = "scratch/clean_cohort_manifests.json"
    with open(out_path, "wb") as f:
        f.write(json.dumps(payload, indent=2).encode('utf-8'))
        
    print(f"Clean manifest written to {out_path} with SHA-256: {payload_hash}")
    print("Recalculated Counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
