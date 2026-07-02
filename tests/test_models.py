"""
test_models.py
---------------
Tests for the failure prediction model (src/models/failure_predictor.py).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.features.test_features import build_feature_matrix
from src.models.failure_predictor import FailurePredictor


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
