"""
baseline.py
-----------
Computes and persists baseline performance statistics (mean, stddev,
percentiles) per endpoint from a trusted historical period.

Used to fit the anomaly detector on known-good data rather than on the same
window being scored - fitting and scoring the same window lets injected
anomalies calibrate their own detector (see anomaly_detector.fit_predict
docstring).
"""

from pathlib import Path

import pandas as pd

BASELINE_METRICS = ["median_ms", "p95_ms", "p99_ms", "rps", "error_rate"]


def compute_baseline_stats(
    history_df: pd.DataFrame, group_col: str = "endpoint"
) -> pd.DataFrame:
    """Compute mean/stddev/percentile baseline stats per endpoint.

    Args:
        history_df: Raw Locust history DataFrame (from locust_parser).
        group_col:  Column to group baseline stats by.

    Returns:
        DataFrame indexed by group_col with {metric}_mean, {metric}_std,
        {metric}_p50, {metric}_p95 for each available metric, plus n_samples.
    """
    available = [m for m in BASELINE_METRICS if m in history_df.columns]
    if not available:
        raise ValueError(
            f"history_df has none of the expected baseline metrics: {BASELINE_METRICS}"
        )

    rows = []
    for group_val, grp in history_df.groupby(group_col):
        row = {group_col: group_val, "n_samples": len(grp)}
        for metric in available:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_std"] = float(grp[metric].std(ddof=0))
            row[f"{metric}_p50"] = float(grp[metric].quantile(0.50))
            row[f"{metric}_p95"] = float(grp[metric].quantile(0.95))
        rows.append(row)

    return pd.DataFrame(rows).set_index(group_col)


def split_baseline_period(
    history_df: pd.DataFrame,
    baseline_fraction: float = 0.3,
    timestamp_col: str = "timestamp",
) -> tuple:
    """Chronologically split history into an early trusted-baseline period
    and the remaining data to be scored against it.

    Using the earliest slice (rather than a random sample) approximates the
    real-world workflow: establish a baseline from known-good historical
    runs, then monitor new runs against it.

    Returns:
        (baseline_df, remainder_df)
    """
    if not 0.0 < baseline_fraction < 1.0:
        raise ValueError("baseline_fraction must be between 0 and 1")

    df = history_df.sort_values(timestamp_col).reset_index(drop=True)
    split_idx = max(1, int(len(df) * baseline_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def save_baseline(stats_df: pd.DataFrame, path: str) -> None:
    """Persist baseline stats to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_json(path, orient="index", indent=2)


def load_baseline(path: str) -> pd.DataFrame:
    """Load previously persisted baseline stats."""
    return pd.read_json(path, orient="index")
