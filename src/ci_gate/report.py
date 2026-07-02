"""
report.py
---------
Generates a self-contained HTML report summarising:
  - Top high-risk tests (failure prediction)
  - Feature importance chart
  - Performance anomaly timeline
  - CI gate decision

The HTML is a single file with embedded CSS — no external dependencies.
Open it directly in a browser or attach it as a GitHub Actions artifact.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QA ML Framework — Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 24px; background: #f8f8f6; color: #1a1a18; }}
  h1   {{ font-size: 22px; font-weight: 600; margin-bottom: 4px; }}
  h2   {{ font-size: 16px; font-weight: 600; color: #0F6E56; margin: 32px 0 12px; }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 32px; }}
  .card {{ background: #fff; border: 1px solid #e2e1da; border-radius: 8px;
           padding: 20px; margin-bottom: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th    {{ background: #f0efe8; text-align: left; padding: 8px 12px;
           font-weight: 600; border-bottom: 1px solid #d3d1c7; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #ebebeb; }}
  tr:hover td {{ background: #fafaf8; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 11px; font-weight: 600; }}
  .badge-high     {{ background: #FAECE7; color: #712B13; }}
  .badge-medium   {{ background: #FAEEDA; color: #633806; }}
  .badge-low      {{ background: #EAF3DE; color: #27500A; }}
  .badge-critical {{ background: #FCEBEB; color: #791F1F; }}
  .badge-warning  {{ background: #FAEEDA; color: #633806; }}
  .badge-normal   {{ background: #EAF3DE; color: #27500A; }}
  .gate-pass {{ background: #EAF3DE; border-left: 4px solid #0F6E56;
                padding: 12px 16px; border-radius: 4px; }}
  .gate-fail {{ background: #FAECE7; border-left: 4px solid #993C1D;
                padding: 12px 16px; border-radius: 4px; }}
  .bar-outer {{ background: #e8e7e0; border-radius: 4px; height: 10px; width: 200px;
                display: inline-block; vertical-align: middle; }}
  .bar-inner {{ height: 10px; border-radius: 4px; background: #0F6E56; }}
</style>
</head>
<body>
<h1>QA ML Framework — Prediction Report</h1>
<p class="meta">Generated: {timestamp} &nbsp;|&nbsp; Commit: {commit_sha}</p>

{gate_block}

<h2>Top High-Risk Tests</h2>
<div class="card">
{risk_table}
</div>

{feature_importance_block}

{anomaly_block}

</body>
</html>"""


def generate_html_report(
    predictions_df: pd.DataFrame,
    gate_result: dict = None,
    feature_importances: dict = None,
    anomaly_df: pd.DataFrame = None,
    commit_sha: str = "unknown",
    output_path: str = "reports/prediction_report.html",
) -> str:
    """Generate a self-contained HTML report.

    Args:
        predictions_df:    DataFrame with test_id, p_fail, risk_rank
        gate_result:       Dict from ci_gate.gate (optional)
        feature_importances: Dict of {feature: importance} (optional)
        anomaly_df:        DataFrame with anomaly detection results (optional)
        commit_sha:        Git commit SHA for the report header
        output_path:       Where to write the HTML file

    Returns:
        Path to the written HTML file.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Gate block
    if gate_result:
        status = gate_result.get("gate_passed", False)
        cls = "gate-pass" if status else "gate-fail"
        icon = "✅" if status else "❌"
        gate_block = f"""
<h2>CI Gate Decision</h2>
<div class="{cls}">
  <strong>{icon} Gate {'PASSED' if status else 'FAILED'}</strong>&nbsp;&nbsp;
  Mean top-risk score: <strong>{gate_result.get('mean_top_risk', 0):.4f}</strong>
  &nbsp;/&nbsp; Threshold: {gate_result.get('risk_threshold', 0):.4f}
  &nbsp;|&nbsp; High-risk tests: {gate_result.get('high_risk_count', 0)}
</div>"""
    else:
        gate_block = ""

    # Risk table
    top = predictions_df.sort_values("p_fail", ascending=False).head(25)
    rows = ""
    for _, r in top.iterrows():
        p = r["p_fail"]
        if p >= 0.6:
            badge = '<span class="badge badge-high">HIGH</span>'
        elif p >= 0.35:
            badge = '<span class="badge badge-medium">MEDIUM</span>'
        else:
            badge = '<span class="badge badge-low">LOW</span>'

        bar_pct = int(p * 100)
        bar = (f'<div class="bar-outer"><div class="bar-inner" '
               f'style="width:{bar_pct}%"></div></div>')
        rows += (f"<tr><td>{int(r.get('risk_rank', 0))}</td>"
                 f"<td>{r.get('test_id', '')}</td>"
                 f"<td>{p:.4f} &nbsp;{bar}</td>"
                 f"<td>{badge}</td></tr>\n")

    risk_table = f"""<table>
<thead><tr><th>#</th><th>Test ID</th><th>P(fail)</th><th>Risk</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""

    # Feature importance block
    if feature_importances:
        fi_rows = ""
        sorted_fi = sorted(feature_importances.items(), key=lambda x: -x[1])
        for feat, imp in sorted_fi:
            bar_pct = int(imp * 100 * 4)  # scale for visibility
            bar = (f'<div class="bar-outer" style="width:300px">'
                   f'<div class="bar-inner" style="width:{min(bar_pct,100)}%"></div></div>')
            fi_rows += (f"<tr><td><code>{feat}</code></td>"
                        f"<td>{imp:.4f} &nbsp;{bar}</td></tr>\n")
        feature_importance_block = f"""
<h2>Feature Importances (Random Forest)</h2>
<div class="card">
<table>
<thead><tr><th>Feature</th><th>Importance</th></tr></thead>
<tbody>{fi_rows}</tbody>
</table>
</div>"""
    else:
        feature_importance_block = ""

    # Anomaly block
    if anomaly_df is not None and not anomaly_df.empty:
        anomalies = anomaly_df[anomaly_df["is_anomaly"]].head(20)
        a_rows = ""
        for _, r in anomalies.iterrows():
            sev = r.get("severity", "warning")
            badge = f'<span class="badge badge-{sev}">{sev.upper()}</span>'
            ts = str(r.get("timestamp", ""))[:19]
            a_rows += (f"<tr><td>{ts}</td>"
                       f"<td>{r.get('endpoint', '')}</td>"
                       f"<td>{r.get('raw_p95_ms', 0):.1f} ms</td>"
                       f"<td>{r.get('raw_error_rate', 0):.3f}</td>"
                       f"<td>{r.get('anomaly_score', 0):.4f}</td>"
                       f"<td>{badge}</td></tr>\n")
        anomaly_block = f"""
<h2>Performance Anomalies</h2>
<div class="card">
<table>
<thead><tr>
  <th>Timestamp</th><th>Endpoint</th><th>P95 (ms)</th>
  <th>Error Rate</th><th>Anomaly Score</th><th>Severity</th>
</tr></thead>
<tbody>{a_rows}</tbody>
</table>
</div>"""
    else:
        anomaly_block = ""

    html = _HTML_TEMPLATE.format(
        timestamp=timestamp,
        commit_sha=commit_sha,
        gate_block=gate_block,
        risk_table=risk_table,
        feature_importance_block=feature_importance_block,
        anomaly_block=anomaly_block,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML report → {output_path}")
    return output_path
