"""
perf_features.py
----------------
Transforms Locust/JMeter time-series data into a feature matrix
for anomaly detection (Isolation Forest).

Each row = one time window for one endpoint.
Features capture both absolute values and deviation from rolling baseline.
"""

import numpy as np
import pandas as pd


def build_perf_feature_matrix(
    history_df: pd.DataFrame,
    baseline_window: int = 50,
) -> pd.DataFrame:
    """Build a feature matrix for performance anomaly detection.

    For each row in history_df, compute deviation-from-baseline features
    using a rolling window of the preceding N samples (no leakage).

    Args:
        history_df:      DataFrame from locust_parser with columns:
                         timestamp, endpoint, median_ms, p95_ms, p99_ms,
                         rps, error_rate (at minimum)
        baseline_window: Number of preceding rows used to compute baseline stats.

    Returns:
        Feature DataFrame with columns:
            timestamp, endpoint,
            raw_median_ms, raw_p95_ms, raw_error_rate, raw_rps,
            z_median_ms, z_p95_ms, z_error_rate, z_rps,
            p95_to_median_ratio, throughput_drop_pct
    """
    required = {"timestamp", "endpoint", "median_ms", "p95_ms", "error_rate", "rps"}
    missing = required - set(history_df.columns)
    if missing:
        raise ValueError(f"history_df missing columns: {missing}")

    df = history_df.copy().sort_values(["endpoint", "timestamp"]).reset_index(drop=True)
    rows = []

    for endpoint, grp in df.groupby("endpoint"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for i, row in grp.iterrows():
            baseline = grp.iloc[max(0, i - baseline_window): i]
            if len(baseline) < 5:
                continue  # Not enough baseline data

            features = {
                "timestamp": row["timestamp"],
                "endpoint": endpoint,
                # ── Raw values ────────────────────────────────────────
                "raw_median_ms": row.get("median_ms", 0.0),
                "raw_p95_ms": row.get("p95_ms", 0.0),
                "raw_p99_ms": row.get("p99_ms", 0.0),
                "raw_error_rate": row.get("error_rate", 0.0),
                "raw_rps": row.get("rps", 0.0),
                # ── Z-scores vs rolling baseline ──────────────────────
                "z_median_ms": _zscore(row.get("median_ms", 0.0),
                                       baseline["median_ms"]),
                "z_p95_ms": _zscore(row.get("p95_ms", 0.0),
                                    baseline["p95_ms"]),
                "z_error_rate": _zscore(row.get("error_rate", 0.0),
                                        baseline["error_rate"]),
                "z_rps": _zscore(row.get("rps", 0.0), baseline["rps"]),
                # ── Ratio features ────────────────────────────────────
                "p95_to_median_ratio": _safe_ratio(row.get("p95_ms", 0.0),
                                                   row.get("median_ms", 1.0)),
                "throughput_drop_pct": _throughput_drop(
                    row.get("rps", 0.0), baseline["rps"]
                ),
            }
            rows.append(features)

    feat_df = pd.DataFrame(rows)
    feat_df.fillna(0.0, inplace=True)
    feat_df.replace([np.inf, -np.inf], 0.0, inplace=True)
    return feat_df


def _zscore(value: float, baseline: pd.Series) -> float:
    """Z-score of value relative to a baseline series."""
    std = baseline.std(ddof=0)
    mean = baseline.mean()
    if std == 0 or pd.isna(std):
        return 0.0
    return (value - mean) / std


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Safe division; returns 0 if denominator is 0."""
    return numerator / denominator if denominator != 0 else 0.0


def _throughput_drop(current_rps: float, baseline_rps: pd.Series) -> float:
    """Percentage drop from baseline mean. Positive = drop, negative = surge."""
    baseline_mean = baseline_rps.mean()
    if baseline_mean == 0:
        return 0.0
    return (baseline_mean - current_rps) / baseline_mean * 100.0


# Feature columns used by the Isolation Forest
PERF_FEATURE_COLUMNS = [
    "z_median_ms",
    "z_p95_ms",
    "z_error_rate",
    "z_rps",
    "p95_to_median_ratio",
    "throughput_drop_pct",
]
