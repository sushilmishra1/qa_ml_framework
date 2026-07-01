"""
test_features.py
----------------
Transforms raw test run history (from junit_parser) into a feature matrix
suitable for scikit-learn classifiers.

Each row in the output represents ONE test at ONE point in time:
    "what features did this test have just before this run, and did it fail?"

This is the core ML problem: predict P(fail) from historical signals.

Key design decisions:
    - Time-aware: features are computed using only PAST data (no leakage)
    - Rolling windows: capture recent trends, not lifetime averages
    - Binary label: failed/error = 1, passed/skipped = 0
"""

import numpy as np
import pandas as pd


def build_feature_matrix(
    history_df: pd.DataFrame,
    rolling_days: int = 30,
    churn_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build the ML feature matrix from test run history.

    For each (test_id, run) pair, compute features using only data
    from BEFORE that run (strict temporal ordering — no leakage).

    Args:
        history_df:  DataFrame from junit_parser with columns:
                     test_id, status, duration_ms, run_timestamp, commit_sha
        rolling_days: Window in days for rolling rate features.
        churn_df:    Optional DataFrame with columns (commit_sha, test_id, overlap_score)
                     representing the files-changed-overlap feature.

    Returns:
        Feature matrix DataFrame with columns:
            test_id, run_timestamp, commit_sha,
            failure_rate_30d, avg_duration_ms, duration_stddev,
            days_since_last_fail, flakiness_score, file_overlap_score,
            is_failed (label — 1=fail/error, 0=pass/skip)
    """
    df = history_df.copy()
    df["is_failed"] = df["status"].isin(["failed", "error"]).astype(int)
    df = df.sort_values("run_timestamp").reset_index(drop=True)

    rows = []
    for test_id, group in df.groupby("test_id"):
        group = group.sort_values("run_timestamp").reset_index(drop=True)
        for i, row in group.iterrows():
            cutoff = row["run_timestamp"]
            past = group[group["run_timestamp"] < cutoff]

            if len(past) < 3:
                # Not enough history to make a reliable prediction
                continue

            window_start = cutoff - pd.Timedelta(days=rolling_days)
            recent = past[past["run_timestamp"] >= window_start]

            features = {
                "test_id": test_id,
                "run_timestamp": cutoff,
                "commit_sha": row["commit_sha"],
                # ── Failure rate features ──────────────────────────────
                "failure_rate_30d": _safe_mean(recent["is_failed"]),
                "failure_rate_all": _safe_mean(past["is_failed"]),
                # ── Duration features (proxy for infrastructure dep) ───
                "avg_duration_ms": past["duration_ms"].mean(),
                "duration_stddev": past["duration_ms"].std(ddof=0),
                "duration_cv": _coeff_variation(past["duration_ms"]),
                # ── Recency features ──────────────────────────────────
                "days_since_last_fail": _days_since_last_fail(past, cutoff),
                "consecutive_passes": _consecutive_passes(past),
                # ── Flakiness score (alternating pass/fail on same SHA) ─
                "flakiness_score": _flakiness_score(past),
                # ── File overlap (from churn_df if provided) ──────────
                "file_overlap_score": _get_overlap(churn_df, test_id,
                                                   row["commit_sha"]),
                # ── Label ──────────────────────────────────────────────
                "is_failed": row["is_failed"],
            }
            rows.append(features)

    feature_df = pd.DataFrame(rows)
    feature_df.fillna(0, inplace=True)
    return feature_df


# ── Feature helper functions ──────────────────────────────────────────────

def _safe_mean(series: pd.Series) -> float:
    """Mean that returns 0.0 on empty series."""
    return float(series.mean()) if len(series) > 0 else 0.0


def _days_since_last_fail(past: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    """Number of days since this test last failed. Returns 999 if never failed."""
    failures = past[past["is_failed"] == 1]
    if failures.empty:
        return 999.0
    last_fail = failures["run_timestamp"].max()
    return (cutoff - last_fail).total_seconds() / 86400.0


def _consecutive_passes(past: pd.DataFrame) -> int:
    """Count consecutive passes at the tail of the history (most recent first)."""
    statuses = past.sort_values("run_timestamp", ascending=False)["is_failed"].tolist()
    count = 0
    for s in statuses:
        if s == 0:
            count += 1
        else:
            break
    return count


def _flakiness_score(past: pd.DataFrame) -> float:
    """
    Flakiness score: fraction of adjacent-run pairs where status alternates.
    A perfectly flaky test (PFPFPF) scores 1.0.
    A perfectly stable test (PPPPPP or FFFFFF) scores 0.0.
    """
    statuses = past.sort_values("run_timestamp")["is_failed"].tolist()
    if len(statuses) < 2:
        return 0.0
    alternations = sum(
        1 for a, b in zip(statuses, statuses[1:]) if a != b
    )
    return alternations / (len(statuses) - 1)


def _coeff_variation(series: pd.Series) -> float:
    """Coefficient of variation (stddev / mean). High CV = unstable duration."""
    if series.mean() == 0 or len(series) < 2:
        return 0.0
    return float(series.std(ddof=0) / series.mean())


def _get_overlap(churn_df: pd.DataFrame, test_id: str,
                 commit_sha: str) -> float:
    """Look up file overlap score from the churn DataFrame."""
    if churn_df is None or churn_df.empty:
        return 0.0
    mask = (churn_df["test_id"] == test_id) & (churn_df["commit_sha"] == commit_sha)
    matches = churn_df[mask]["overlap_score"]
    return float(matches.iloc[0]) if not matches.empty else 0.0


# ── Feature column sets (used by model training) ─────────────────────────

FEATURE_COLUMNS = [
    "failure_rate_30d",
    "failure_rate_all",
    "avg_duration_ms",
    "duration_stddev",
    "duration_cv",
    "days_since_last_fail",
    "consecutive_passes",
    "flakiness_score",
    "file_overlap_score",
]

LABEL_COLUMN = "is_failed"
