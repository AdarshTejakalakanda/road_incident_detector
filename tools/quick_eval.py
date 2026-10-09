import os, sys
sys.path.insert(0, os.path.abspath('.'))
import tools.evaluate_wrong_way_target as e

cfg = e.load_config('road_surveillance/config/default_config.yaml')
yolo_model = e.YOLO(cfg.detection.model_path)
preprocessor = e.LowLightPreprocessor(mode='N3')
vocab_gate = e.VocabularyGate()
quality_gate = e.DetectorQualityGate(tau_high=0.25, tau_low=0.10, vocab_gate=vocab_gate)
motion_gate = e.MotionROIGate()
arbitrator = e.GroqArbitrator(mock_fallback=True)

res = e.run_pipeline_on_video(
    video_path=r'C:\Users\KALAKA~1\Downloads\archive\real_videos\KTGByI3N4G0_00.mp4',
    video_id='KTGByI3N4G0_00',
    gt_frame=91,
    cfg=cfg,
    yolo_model=yolo_model,
    preprocessor=preprocessor,
    vocab_gate=vocab_gate,
    quality_gate=quality_gate,
    motion_gate=motion_gate,
    arbitrator=arbitrator,
    max_frames=130,
)

print(f"Raw candidates count: {res['raw_candidates_count']}")
for c in res['raw_candidates']:
    print(f"  F{c['frame']} | {c['event_type']} | tracks={c['track_ids']} | {c['description']}")

print(f"\nConsolidated episodes count: {res['consolidated_episodes_count']}")
for i, ep in enumerate(res['consolidated_episodes']):
    print(f"  Ep {i+1}: {ep.primary_event_type} | F{ep.start_frame}->F{ep.end_frame} (Peak F{ep.peak_frame}) | tracks={ep.involved_track_ids} | in_gt={ep.is_in_gt_window} | cands={len(ep.raw_candidates)} | near_misses={len(ep.attached_near_misses)}")

print(f"\nSuppressed events count: {len(res['suppressed_events'])}")
from collections import Counter
counts = Counter(s['reason'] for s in res['suppressed_events'])
for r, cnt in counts.items():
    print(f"  {r}: {cnt}")
