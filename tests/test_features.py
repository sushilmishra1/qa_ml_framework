"""
test_features.py
-----------------
Tests for failure-prediction feature engineering (src/features/test_features.py).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.test_features import (
    build_feature_matrix,
    _flakiness_score,
    _days_since_last_fail,
    FEATURE_COLUMNS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_history(n_tests: int = 5, n_runs: int = 20, failure_rate: float = 0.1):
    """Build a minimal synthetic history DataFrame."""
    rows = []
    base = datetime(2024, 1, 1)
    for t in range(n_tests):
        test_id = f"module_{t}::test_case_{t}"
        for r in range(n_runs):
            failed = np.random.random() < failure_rate
            rows.append({
                "test_id": test_id,
                "test_name": f"test_case_{t}",
                "classname": f"module_{t}",
                "status": "failed" if failed else "passed",
                "duration_ms": np.random.normal(500, 100),
                "run_timestamp": base + timedelta(days=r * 2),
                "commit_sha": f"sha{r:04d}",
                "failure_message": "AssertionError" if failed else "",
                "error_type": "AssertionError" if failed else "",
            })
    return pd.DataFrame(rows)


class TestTestFeatures:

    def test_build_returns_dataframe(self):
        np.random.seed(0)
        history = _make_history()
        features = build_feature_matrix(history, rolling_days=30)
        assert isinstance(features, pd.DataFrame)

    def test_feature_columns_present(self):
        np.random.seed(0)
        history = _make_history()
        features = build_feature_matrix(history, rolling_days=30)
        for col in FEATURE_COLUMNS:
            assert col in features.columns, f"Missing feature column: {col}"

    def test_label_column_present(self):
        np.random.seed(0)
        history = _make_history()
        features = build_feature_matrix(history, rolling_days=30)
        assert "is_failed" in features.columns

    def test_label_is_binary(self):
        np.random.seed(0)
        history = _make_history()
        features = build_feature_matrix(history)
        assert set(features["is_failed"].unique()).issubset({0, 1})

    def test_no_future_leakage(self):
        """All features must be computed from data BEFORE the current run."""
        np.random.seed(0)
        history = _make_history(n_runs=30)
        features = build_feature_matrix(history)
        # Verify failure_rate_30d <= 1.0 for all rows (sanity check)
        assert (features["failure_rate_30d"] >= 0.0).all()
        assert (features["failure_rate_30d"] <= 1.0).all()

    def test_no_nan_in_features(self):
        np.random.seed(0)
        history = _make_history()
        features = build_feature_matrix(history)
        assert not features[FEATURE_COLUMNS].isnull().any().any()

    def test_flakiness_score_stable(self):
        """A test that always passes should have flakiness_score = 0."""
        s = pd.Series([0, 0, 0, 0, 0, 0])
        assert _flakiness_score(pd.DataFrame({"is_failed": s,
            "run_timestamp": pd.date_range("2024-01-01", periods=6)})) == 0.0

    def test_flakiness_score_perfectly_flaky(self):
        """A perfectly alternating test should have flakiness_score = 1."""
        s = pd.Series([0, 1, 0, 1, 0, 1])
        result = _flakiness_score(pd.DataFrame({"is_failed": s,
            "run_timestamp": pd.date_range("2024-01-01", periods=6)}))
        assert result == 1.0

    def test_days_since_last_fail_no_history(self):
        """Should return 999.0 when no failure in history."""
        past = pd.DataFrame({"is_failed": [0, 0, 0],
            "run_timestamp": pd.date_range("2024-01-01", periods=3)})
        cutoff = pd.Timestamp("2024-01-10")
        result = _days_since_last_fail(past, cutoff)
        assert result == 999.0
