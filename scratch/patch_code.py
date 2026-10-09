with open('modules/vlm_arbitrator.py', 'r', encoding='utf-8') as f:
    orig = f.read()
idx = orig.find('def _evaluate_deterministic_high_confidence_bypass')
idx_end = orig.find('def _evaluate_and_dispatch_fallback')
open('scratch/current_bypass.py', 'w', encoding='utf-8').write(orig[idx:idx_end])
print('Saved current bypass implementation: len', idx_end - idx)