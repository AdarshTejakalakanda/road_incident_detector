import json
import numpy as np

# Load the 53-episode evaluation audit
with open('artifacts/aftermath_evaluation_audit_5uh8-3q51uI_00.json', 'r') as f:
    eval_audit = json.load(f)

# Load the 16 quarantined suppression audit
with open('artifacts/aftermath_suppression_audit_5uh8-3q51uI_00.json', 'r') as f:
    supp_audit = json.load(f)

supp_ids = set(e['episode_id'] for e in supp_audit)
f176_id = '5uh8-3q51uI_00_ep01_f24'

# Extract F176
f176_ep = [ep for ep in eval_audit if ep['episode_id'] == f176_id][0]

# Define the 4 Crossing TPs (Historical Type 3)
crossing_tps = [
    {
        'id': 'unS0-TLF1ao_00',
        'frame': 47,
        'type': 'Type 3',
        'N': 4,
        'phi_dv': 1.0,
        'phi_rest': 1.0,
        'phi_rot': 1.0,
        'phi_ovl': 1.0,
        'phi_lat': 1.0,
        'phi_push': 1.0,
        'phi_up_binary': 1.0,
        'baseline_score': 1.0000,
    },
    {
        'id': 'kABWTDKgc9U_00',
        'frame': 242,
        'type': 'Type 3',
        'N': 5,
        'phi_dv': 1.0,
        'phi_rest': 0.85,
        'phi_rot': 1.0,
        'phi_ovl': 0.90,
        'phi_lat': 0.925,
        'phi_push': 1.0,
        'phi_up_binary': 1.0,
        'baseline_score': 0.9600,
    },
    {
        'id': 'ftk_fSc4haM_00',
        'frame': 121,
        'type': 'Type 3',
        'N': 4,
        'phi_dv': 1.0,
        'phi_rest': 0.85,
        'phi_rot': 1.0,
        'phi_ovl': 0.90,
        'phi_lat': 0.97,
        'phi_push': 0.875,
        'phi_up_binary': 1.0,
        'baseline_score': 0.9520,
    },
    {
        'id': 'fzWY0vLAXzI_00',
        'frame': 131,
        'type': 'Type 3',
        'N': 3,
        'phi_dv': 1.0,
        'phi_rest': 1.0,
        'phi_rot': 1.0,
        'phi_ovl': 1.0,
        'phi_lat': 0.90,
        'phi_push': 0.90,
        'phi_up_binary': 1.0,
        'baseline_score': 0.9570,
    }
]

# Define the 4 5uh8 crossing episodes mentioned in step 138
crossing_5uh8_ids = ['5uh8-3q51uI_00_ep27_f467', '5uh8-3q51uI_00_ep31_f493', '5uh8-3q51uI_00_ep33_f530', '5uh8-3q51uI_00_ep38_f609']
crossing_5uh8 = [ep for ep in eval_audit if ep['episode_id'] in crossing_5uh8_ids]

# Type weights
WEIGHTS = {
    'Type 1': {'w_v': 0.30, 'w_rest': 0.15, 'w_rot': 0.10, 'w_ovl': 0.15, 'w_lat': 0.15, 'w_push': 0.05, 'w_up': 0.10},
    'Type 2': {'w_v': 0.15, 'w_rest': 0.30, 'w_rot': 0.05, 'w_ovl': 0.20, 'w_lat': 0.15, 'w_push': 0.05, 'w_up': 0.10},
    'Type 3': {'w_v': 0.20, 'w_rest': 0.15, 'w_rot': 0.30, 'w_ovl': 0.10, 'w_lat': 0.10, 'w_push': 0.10, 'w_up': 0.05},
    'Type 4': {'w_v': 0.15, 'w_rest': 0.10, 'w_rot': 0.15, 'w_ovl': 0.20, 'w_lat': 0.30, 'w_push': 0.05, 'w_up': 0.05},
    'Type 5': {'w_v': 0.30, 'w_rest': 0.30, 'w_rot': 0.25, 'w_ovl': 0.00, 'w_lat': 0.00, 'w_push': 0.00, 'w_up': 0.15},
}

def compute_ep_score(channels, ctype, phi_up_override=None):
    w = WEIGHTS[ctype]
    up = phi_up_override if phi_up_override is not None else channels['phi_up']
    return (
        w['w_v'] * channels['phi_dv'] +
        w['w_rest'] * channels['phi_rest'] +
        w['w_rot'] * channels['phi_rot'] +
        w['w_ovl'] * channels['phi_ovl'] +
        w['w_lat'] * channels['phi_lat'] +
        w['w_push'] * channels['phi_push'] +
        w['w_up'] * up
    )

print("=== TRACK 2: GRADED PHI_UP SWEEP ===")
k_values = [1, 2, 5, 10, 25, 30, 50]

# F176 evaluation across k
print("\n--- F176 Evaluation ---")
f176_ch = f176_ep['channels']
f176_N = f176_ep['telemetry']['upstream_react']

for k in k_values:
    phi_up_k = min(1.0, f176_N / k)
    # Under Type 3 (Pipeline Canonical Type) -> E_P1
    e_p1 = compute_ep_score(f176_ch, 'Type 3', phi_up_override=phi_up_k)
    # Under Type 1 (Ground Truth Highway Collision) -> E_P2
    e_p2 = compute_ep_score(f176_ch, 'Type 1', phi_up_override=phi_up_k)
    # Under Type 5 (Single Vehicle Spin/Barrier) -> E_Type5
    e_t5 = compute_ep_score(f176_ch, 'Type 5', phi_up_override=phi_up_k)
    print(f"k={k:2d}: phi_up={phi_up_k:.4f} | E_P1(Type 3)={e_p1:.4f} | E_P2(Type 1)={e_p2:.4f} | E(Type 5)={e_t5:.4f}")

# The 4 Historical Crossing TPs across k
print("\n--- 4 Historical Crossing TPs (Set A) ---")
for ep in crossing_tps:
    print(f"\nEpisode {ep['id']} (N={ep['N']}):")
    for k in [1, 2, 5, 10]:
        phi_up_k = min(1.0, ep['N'] / k)
        score = ep['baseline_score'] - WEIGHTS['Type 3']['w_up'] * (1.0 - phi_up_k)
        survives = score >= 0.50
        print(f"  k={k:2d}: phi_up={phi_up_k:.4f} | score={score:.4f} | survives={survives}")

# The 4 5uh8 crossing episodes (Set B) across k
print("\n--- 4 5uh8 Episodes (Set B) ---")
for ep in crossing_5uh8:
    eid = ep['episode_id']
    ctype = ep['canonical_type']
    N = ep['telemetry']['upstream_react']
    base_s = ep['score']
    print(f"\nEpisode {eid} (Type={ctype}, N={N}, base={base_s:.4f}):")
    for k in [1, 2, 5, 10, 25, 30, 50]:
        phi_up_k = min(1.0, N / k)
        s = compute_ep_score(ep['channels'], ctype, phi_up_override=phi_up_k)
        survives = s >= 0.50
        print(f"  k={k:2d}: phi_up={phi_up_k:.4f} | score={s:.4f} | survives={survives}")

# Full 53-episode cohort distribution across k
print("\n--- Full 53-Episode Cohort Score Distribution ---")
for k in [1, 2, 5, 10, 25, 30, 50]:
    scores = []
    for ep in eval_audit:
        if ep['score'] is None:
            continue
        ctype = ep['canonical_type']
        N = ep['telemetry'].get('upstream_react', 0)
        phi_up_k = min(1.0, N / k)
        s = compute_ep_score(ep['channels'], ctype, phi_up_override=phi_up_k)
        scores.append(s)
    scores = np.array(scores)
    escalated_cnt = sum(scores >= 0.50)
    suppressed_cnt = sum(scores < 0.50)
    print(f"k={k:2d}: min={scores.min():.4f}, median={np.median(scores):.4f}, max={scores.max():.4f}, mean={scores.mean():.4f} | Escalated={escalated_cnt}, Suppressed={suppressed_cnt}")
