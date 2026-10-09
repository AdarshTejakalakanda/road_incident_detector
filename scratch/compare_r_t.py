import json

with open('artifacts/r_full_benchmark_sweep.json') as f:
    r = json.load(f)
with open('artifacts/t_full_benchmark_sweep.json') as f:
    t = json.load(f)

r_dict = {(x['video_id'], x['split_type']): x for x in r['per_video_results']}
t_dict = {(x['video_id'], x['split_type']): x for x in t['per_video_results']}

print(f"{'#':<2} | {'Video ID':<16} | {'Split':<22} | {'R TP':<5} | {'T TP':<5} | {'R FP':<4} | {'T FP':<4} | Notes")
print("-" * 80)
idx = 1
for k, r_res in r_dict.items():
    t_res = t_dict.get(k, {})
    r_tp = r_res.get('is_tp', False)
    t_tp = t_res.get('is_tp', False)
    r_fp = r_res.get('fp_confirmed_count', 0)
    t_fp = t_res.get('fp_confirmed_count', 0)
    note = ""
    if not r_tp and t_tp:
        note = "RECOVERED TP in T"
    elif r_tp and not t_tp:
        note = "REGRESSION in T"
    elif r_fp != t_fp:
        note = f"FP change: {r_fp} -> {t_fp}"
    print(f"{idx:02d} | {k[0]:<16} | {k[1]:<22} | {str(r_tp):<5} | {str(t_tp):<5} | {r_fp:<4} | {t_fp:<4} | {note}")
    idx += 1
