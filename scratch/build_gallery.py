import os
import glob
import csv

base_dir = "evidence/threshold_sweep_yzNEliTlFGk"
pkg_dir = os.path.join(base_dir, "packages")
sheets_dir = os.path.join(base_dir, "verification_sheets")
out_html = os.path.join(base_dir, "gallery.html")

# Read metadata from all verification CSVs
alert_metadata = {}
for sheet_file in glob.glob(os.path.join(sheets_dir, "*.csv")):
    sheet_name = os.path.basename(sheet_file)
    # verification_{arm}_tau{tau_str}.csv
    parts = sheet_name.replace("verification_", "").replace(".csv", "").split("_tau")
    if len(parts) == 2:
        arm = parts[0]
        tau_str = parts[1]
        with open(sheet_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a_idx = row["alert_idx"]
                c_type = row["esc_type"].replace(" ", "_")
                # Pattern: {arm}_tau{tau_str}_{clip_id}_alert{idx}_{type}
                key = f"{arm}_tau{tau_str}_{row['clip_id']}_alert{a_idx}_{c_type}"
                alert_metadata[key] = row

gifs = sorted(glob.glob(os.path.join(pkg_dir, "*_annotated.gif")))

html_cards = []
for gif in gifs:
    fname = os.path.basename(gif)
    base = fname.replace("_annotated.gif", "")
    filmstrip = os.path.join(pkg_dir, base + "_filmstrip.png")
    ann_mp4 = os.path.join(pkg_dir, base + "_annotated.mp4")
    raw_mp4 = os.path.join(pkg_dir, base + "_raw.mp4")

    meta = alert_metadata.get(base, {})

    rel_gif = "packages/" + os.path.basename(gif)
    rel_filmstrip = "packages/" + os.path.basename(filmstrip) if os.path.exists(filmstrip) else ""
    rel_ann_mp4 = "packages/" + os.path.basename(ann_mp4)
    rel_raw_mp4 = "packages/" + os.path.basename(raw_mp4)

    card = f"""
    <div class="alert-card">
        <div class="card-header">
            <h3>{base}</h3>
            <span class="badge">Escalation Score: {meta.get('esc_score', 'N/A')}</span>
            <span class="badge type">{meta.get('esc_type', 'Incident')}</span>
            <span class="badge frame">Frame {meta.get('frame', 'N/A')} ({meta.get('timestamp_s', 'N/A')}s)</span>
        </div>
        <div class="media-row">
            <div class="gif-box">
                <h4>Dynamic Annotated Animation (Browser Playable)</h4>
                <img src="{rel_gif}" alt="{base}" loading="lazy" />
            </div>
            <div class="filmstrip-box">
                <h4>5-Frame Key Progression Filmstrip</h4>
                <a href="{rel_filmstrip}" target="_blank">
                    <img src="{rel_filmstrip}" alt="filmstrip" loading="lazy" />
                </a>
            </div>
        </div>
        <div class="card-footer">
            <span><b>Contributing Kinematic Channels:</b> {meta.get('contributing_channels', 'None')}</span>
            <div class="file-links">
                <a href="{rel_ann_mp4}" download>Download Annotated MP4</a> &bull;
                <a href="{rel_raw_mp4}" download>Download Raw MP4</a> &bull;
                <a href="{rel_filmstrip}" target="_blank">View Filmstrip HD</a>
            </div>
        </div>
    </div>
    """
    html_cards.append(card)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Road Incident Detector - Visual Adjudication Gallery</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
        margin: 0;
        padding: 24px;
        line-height: 1.5;
    }}
    h1 {{ color: #38bdf8; margin-bottom: 6px; }}
    .subtitle {{ color: #94a3b8; margin-bottom: 24px; font-size: 15px; }}
    .stats-banner {{
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
        display: flex;
        gap: 20px;
        border: 1px solid #334155;
    }}
    .stat-item {{ flex: 1; text-align: center; }}
    .stat-value {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
    .stat-label {{ font-size: 13px; color: #94a3b8; }}
    .alert-card {{
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        margin-bottom: 28px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    .card-header {{
        background: #334155;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }}
    .card-header h3 {{ margin: 0; font-size: 15px; font-family: monospace; color: #f8fafc; flex-grow: 1; }}
    .badge {{
        background: #0284c7;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }}
    .badge.type {{ background: #d97706; }}
    .badge.frame {{ background: #475569; }}
    .media-row {{
        display: flex;
        flex-direction: column;
        gap: 16px;
        padding: 18px;
    }}
    @media (min-width: 900px) {{
        .media-row {{ flex-direction: row; }}
    }}
    .gif-box, .filmstrip-box {{
        flex: 1;
        background: #0b1120;
        border-radius: 6px;
        padding: 12px;
        border: 1px solid #1e293b;
    }}
    .gif-box h4, .filmstrip-box h4 {{
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 13px;
        color: #94a3b8;
    }}
    img {{
        width: 100%;
        height: auto;
        border-radius: 4px;
        display: block;
    }}
    .card-footer {{
        background: #182234;
        padding: 12px 18px;
        font-size: 13px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        border-top: 1px solid #334155;
    }}
    .file-links a {{
        color: #38bdf8;
        text-decoration: none;
        font-size: 13px;
        font-weight: 500;
    }}
    .file-links a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
    <h1>Road Incident Detection - Visual Adjudication Gallery</h1>
    <div class="subtitle">Generated for nocturnal collision clip: <code>yzNEliTlFGk_00.mp4</code> | 21 Escalated Alert Packages</div>
    <div class="stats-banner">
        <div class="stat-item"><div class="stat-value">21</div><div class="stat-label">Total Escalated Alerts</div></div>
        <div class="stat-item"><div class="stat-value">Cand_B (RT-DETRv2-S)</div><div class="stat-label">15 Alerts Across Sweeps</div></div>
        <div class="stat-item"><div class="stat-value">Cand_C (Highway v8m)</div><div class="stat-label">6 Alerts Across Sweeps</div></div>
        <div class="stat-item"><div class="stat-value">Control, A, D</div><div class="stat-label">0 Alerts (Perception Blackout)</div></div>
    </div>
    {''.join(html_cards)}
</body>
</html>
"""

with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Gallery successfully created at: {out_html}")
