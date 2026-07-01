"""
failure_predictor.py
--------------------
Test failure prediction using Random Forest (primary) and
Logistic Regression (interpretable baseline).

Key design decisions:
    - Time-based train/test split (NOT random k-fold — avoids temporal leakage)
    - class_weight='balanced' to handle the ~5% failure rate imbalance
    - Feature importance output for interview/stakeholder explainability
    - Model persistence via joblib for CI gate reuse
"""

import os
import joblib
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.test_features import FEATURE_COLUMNS, LABEL_COLUMN


class FailurePredictor:
    """Predicts test failure probability from historical feature vectors.

    Usage:
        predictor = FailurePredictor(model_type='random_forest')
        metrics = predictor.train(feature_df, split_days=14)
        predictions = predictor.predict(new_feature_df)
        predictor.save('models/failure_predictor.pkl')
    """

    SUPPORTED_MODELS = ("random_forest", "logistic_regression")

    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.random_state = random_state
        self._pipeline = self._build_pipeline(model_type, n_estimators, random_state)
        self._is_trained = False
        self.feature_importances_ = None

    # ── Public API ─────────────────────────────────────────────────────────

    def train(self, feature_df: pd.DataFrame, split_days: int = 14) -> dict:
        """Train the model using a time-based split.

        Args:
            feature_df: Feature matrix from test_features.build_feature_matrix().
            split_days: Hold out the last N days as the test set.

        Returns:
            dict of evaluation metrics.
        """
        df = feature_df.copy()
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])

        cutoff = df["run_timestamp"].max() - timedelta(days=split_days)
        train_df = df[df["run_timestamp"] < cutoff]
        test_df = df[df["run_timestamp"] >= cutoff]

        if train_df.empty or test_df.empty:
            raise ValueError(
                f"Time split produced empty train or test set. "
                f"Reduce split_days (currently {split_days}) or provide more history."
            )

        X_train = train_df[FEATURE_COLUMNS].values
        y_train = train_df[LABEL_COLUMN].values
        X_test = test_df[FEATURE_COLUMNS].values
        y_test = test_df[LABEL_COLUMN].values

        self._pipeline.fit(X_train, y_train)
        self._is_trained = True

        # Extract feature importances for Random Forest
        if self.model_type == "random_forest":
            rf = self._pipeline.named_steps["classifier"]
            self.feature_importances_ = dict(
                zip(FEATURE_COLUMNS, rf.feature_importances_)
            )

        # Evaluate
        y_pred = self._pipeline.predict(X_test)
        y_proba = self._pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model_type": self.model_type,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "split_cutoff": str(cutoff.date()),
            "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "positive_rate_train": round(y_train.mean(), 4),
            "positive_rate_test": round(y_test.mean(), 4),
        }

        print(f"\n{'='*50}")
        print(f"  {self.model_type.replace('_', ' ').title()} — Evaluation")
        print(f"{'='*50}")
        for k, v in metrics.items():
            print(f"  {k:<30} {v}")
        print(classification_report(y_test, y_pred,
                                    target_names=["passed", "failed"],
                                    zero_division=0))

        if self.feature_importances_:
            print("\n  Feature Importances:")
            for feat, imp in sorted(self.feature_importances_.items(),
                                    key=lambda x: -x[1]):
                bar = "█" * int(imp * 40)
                print(f"  {feat:<30} {imp:.4f}  {bar}")

        return metrics

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Predict failure probability for a set of tests.

        Args:
            feature_df: Must contain the FEATURE_COLUMNS columns.

        Returns:
            DataFrame with test_id, p_fail (probability), risk_rank columns.
        """
        self._check_trained()
        X = feature_df[FEATURE_COLUMNS].fillna(0).values
        proba = self._pipeline.predict_proba(X)[:, 1]

        result = feature_df[["test_id"]].copy() if "test_id" in feature_df.columns \
            else pd.DataFrame({"test_id": range(len(feature_df))})
        result["p_fail"] = proba
        result.sort_values("p_fail", ascending=False, inplace=True)
        result["risk_rank"] = range(1, len(result) + 1)
        result.reset_index(drop=True, inplace=True)
        return result

    def save(self, path: str) -> None:
        """Persist model pipeline to disk."""
        self._check_trained()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._pipeline, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FailurePredictor":
        """Load a previously saved model."""
        instance = cls.__new__(cls)
        instance._pipeline = joblib.load(path)
        instance._is_trained = True
        instance.feature_importances_ = None
        instance.model_type = "loaded"
        return instance

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call .train() first.")

    @staticmethod
    def _build_pipeline(
        model_type: str, n_estimators: int, random_state: int
    ) -> Pipeline:
        if model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                max_depth=12,
                min_samples_leaf=5,
            )
            # RF doesn't need scaling, but we include it for pipeline consistency
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ])
        elif model_type == "logistic_regression":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=1000,
                    solver="lbfgs",
                )),
            ])
        raise ValueError(f"Unknown model_type: {model_type}")
