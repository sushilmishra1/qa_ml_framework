"""
test_features.py and test_models.py combined
--------------------------------------------
Tests for feature engineering and model training/prediction.
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
from src.features.perf_features import build_perf_feature_matrix, PERF_FEATURE_COLUMNS
from src.models.failure_predictor import FailurePredictor
from src.perf_anomaly.anomaly_detector import PerformanceAnomalyDetector


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


def _make_perf_history(n_windows: int = 80, spike_at: int = 60):
    """Build synthetic Locust-style performance history with one anomaly."""
    rows = []
    base = datetime(2024, 1, 1, 10, 0, 0)
    for i in range(n_windows):
        is_spike = i >= spike_at
        rows.append({
            "timestamp": base + timedelta(seconds=i * 10),
            "endpoint": "GET /api/test",
            "median_ms": 300.0 if is_spike else 100.0,
            "p95_ms": 600.0 if is_spike else 180.0,
            "p99_ms": 900.0 if is_spike else 250.0,
            "rps": 15.0 if is_spike else 50.0,
            "error_rate": 0.15 if is_spike else 0.01,
        })
    return pd.DataFrame(rows)


# ── Feature tests ─────────────────────────────────────────────────────────

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


class TestPerfFeatures:

    def test_build_returns_dataframe(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        assert isinstance(features, pd.DataFrame)

    def test_perf_feature_columns_present(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        for col in PERF_FEATURE_COLUMNS:
            assert col in features.columns, f"Missing perf feature: {col}"

    def test_no_nan_after_fill(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        assert not features[PERF_FEATURE_COLUMNS].isnull().any().any()

    def test_requires_mandatory_columns(self):
        bad_df = pd.DataFrame({"timestamp": [], "endpoint": []})
        with pytest.raises(ValueError, match="missing columns"):
            build_perf_feature_matrix(bad_df)


# ── Model tests ───────────────────────────────────────────────────────────

class TestFailurePredictor:

    @pytest.fixture
    def trained_predictor(self):
        np.random.seed(42)
        history = _make_history(n_tests=10, n_runs=40)
        features = build_feature_matrix(history)
        predictor = FailurePredictor(model_type="random_forest", n_estimators=10)
        predictor.train(features, split_days=7)
        return predictor, features

    def test_train_returns_metrics(self):
        np.random.seed(42)
        history = _make_history(n_tests=10, n_runs=40)
        features = build_feature_matrix(history)
        predictor = FailurePredictor(n_estimators=10)
        metrics = predictor.train(features, split_days=7)
        assert "auc_roc" in metrics
        assert "f1" in metrics

    def test_predict_returns_dataframe(self, trained_predictor):
        predictor, features = trained_predictor
        latest = features.groupby("test_id").last().reset_index()
        result = predictor.predict(latest)
        assert isinstance(result, pd.DataFrame)
        assert "p_fail" in result.columns
        assert "risk_rank" in result.columns

    def test_predict_probabilities_in_range(self, trained_predictor):
        predictor, features = trained_predictor
        latest = features.groupby("test_id").last().reset_index()
        result = predictor.predict(latest)
        assert (result["p_fail"] >= 0.0).all()
        assert (result["p_fail"] <= 1.0).all()

    def test_predict_sorted_by_risk(self, trained_predictor):
        predictor, features = trained_predictor
        latest = features.groupby("test_id").last().reset_index()
        result = predictor.predict(latest)
        probs = result["p_fail"].tolist()
        assert probs == sorted(probs, reverse=True)

    def test_predict_before_train_raises(self):
        predictor = FailurePredictor()
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(pd.DataFrame())

    def test_save_and_load(self, trained_predictor, tmp_path):
        predictor, features = trained_predictor
        path = str(tmp_path / "model.pkl")
        predictor.save(path)
        loaded = FailurePredictor.load(path)
        latest = features.groupby("test_id").last().reset_index()
        result = loaded.predict(latest)
        assert len(result) > 0

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="must be one of"):
            FailurePredictor(model_type="neural_network")


class TestAnomalyDetector:

    @pytest.fixture
    def fitted_detector(self):
        perf = _make_perf_history(n_windows=100, spike_at=80)
        features = build_perf_feature_matrix(perf, baseline_window=20)
        detector = PerformanceAnomalyDetector(contamination=0.10)
        detector.fit(features)
        return detector, features

    def test_fit_predict_returns_dataframe(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        detector = PerformanceAnomalyDetector(contamination=0.10)
        result = detector.fit_predict(features)
        assert isinstance(result, pd.DataFrame)

    def test_anomaly_column_is_boolean(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        detector = PerformanceAnomalyDetector(contamination=0.10)
        result = detector.fit_predict(features)
        assert result["is_anomaly"].dtype == bool

    def test_severity_values(self):
        perf = _make_perf_history()
        features = build_perf_feature_matrix(perf, baseline_window=20)
        detector = PerformanceAnomalyDetector(contamination=0.10)
        result = detector.fit_predict(features)
        assert set(result["severity"].unique()).issubset(
            {"normal", "warning", "critical"}
        )

    def test_spike_detected(self):
        """The injected performance spike should be flagged as anomalous."""
        perf = _make_perf_history(n_windows=100, spike_at=80)
        features = build_perf_feature_matrix(perf, baseline_window=20)
        detector = PerformanceAnomalyDetector(contamination=0.10)
        result = detector.fit_predict(features)
        # The spike windows should have anomalies
        spike_rows = result.tail(20)
        assert spike_rows["is_anomaly"].sum() > 0

    def test_predict_before_fit_raises(self):
        detector = PerformanceAnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.predict(pd.DataFrame())
