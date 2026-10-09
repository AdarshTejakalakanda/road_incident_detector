import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd

meta = pd.read_csv(r'C:\Users\KALAKA~1\Downloads\archive\metadata-real.csv')

with open('telemetry_iteration_c_single_vehicle.json') as f:
    d = json.load(f)

for split_name in ['split_in_distribution', 'split_geo_aware']:
    print(f"\n========================================================")
    print(f"SPLIT: {split_name}")
    print(f"========================================================")
    diag_b = {x['video_id']: x for x in d[split_name]['iteration_b']['diagnostics']}
    diag_c = {x['video_id']: x for x in d[split_name]['iteration_c_standalone']['diagnostics']}
    diag_comb = {x['video_id']: x for x in d[split_name]['iteration_c_combined']['diagnostics']}

    for vid in diag_b:
        sub = meta[meta['path'].str.contains(vid)]
        row = sub.iloc[0] if len(sub) > 0 else None
        v_type = row['type'] if row is not None else 'unknown'
        roll = row['rollover'] if row is not None else 'unknown'
        gt = diag_b[vid]['gt_frame']
        st_b = diag_b[vid]['status']
        st_c = diag_c[vid]['status']
        st_comb = diag_comb[vid]['status']
        
        # print details
        print(f"{vid:18s} | type={v_type:10s} | roll={roll} | GT={gt:3d} | B={st_b:2s} | C={st_c:2s} | Comb={st_comb:2s} | B_cands={[c['f'] for c in diag_b[vid]['candidates']]} | C_cands={[c['f'] for c in diag_c[vid]['candidates']]}")
