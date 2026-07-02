"""
anomaly_detector.py
-------------------
Detects performance anomalies in Locust/JMeter test results using:
    1. Isolation Forest (multivariate — response time + throughput + errors)
    2. Z-score threshold (simple, explainable single-metric alerts)

Isolation Forest is ideal for performance testing because:
    - No labelled anomaly dataset required (unsupervised)
    - Handles multivariate correlations (slow + high error_rate together)
    - Scales well to large time-series (O(n log n))
    - sklearn IsolationForest is the industry standard implementation

Interview explanation:
    "Anomalies are rare and different — they are easier to ISOLATE than
    normal points. IsolationForest builds random trees and measures how
    few splits it takes to isolate a point. Anomalies need fewer splits
    because they sit in sparse regions of feature space."
"""

import os
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.perf_features import PERF_FEATURE_COLUMNS
from src.perf_anomaly.baseline import split_baseline_period


class PerformanceAnomalyDetector:
    """Detects anomalous time windows in performance test results.

    Usage:
        detector = PerformanceAnomalyDetector(contamination=0.05)
        results = detector.fit_predict(perf_feature_df)
        anomalies = results[results['is_anomaly'] == True]
    """

    def __init__(
        self,
        contamination: float = 0.05,
        zscore_threshold: float = 3.0,
        random_state: int = 42,
    ):
        """
        Args:
            contamination:    Expected fraction of anomalies (0.01–0.5).
                              0.05 = assume 5% of windows are anomalous.
            zscore_threshold: Simple Z-score cutoff for single-metric alerts.
            random_state:     Reproducibility seed.
        """
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=random_state,
                n_jobs=-1,
            )),
        ])
        self._is_fitted = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, feature_df: pd.DataFrame) -> "PerformanceAnomalyDetector":
        """Fit the Isolation Forest on a baseline set of performance data.

        Args:
            feature_df: Feature matrix from perf_features.build_perf_feature_matrix().
        """
        X = feature_df[PERF_FEATURE_COLUMNS].fillna(0).values
        self._pipeline.fit(X)
        self._is_fitted = True
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies for new performance windows.

        Returns:
            DataFrame with original columns plus:
                is_anomaly     : bool   - True if Isolation Forest flags this window
                anomaly_score  : float  - raw anomaly score (more negative = more anomalous)
                zscore_p95     : float  - Z-score of p95 response time
                zscore_alert   : bool   - True if any Z-score exceeds threshold
                severity       : str    - 'critical' | 'warning' | 'normal'
        """
        self._check_fitted()
        df = feature_df.copy()
        X = df[PERF_FEATURE_COLUMNS].fillna(0).values

        # Isolation Forest: -1 = anomaly, 1 = normal
        raw_labels = self._pipeline.predict(X)
        # Score: more negative = more anomalous
        scores = self._pipeline.decision_function(X)

        df["is_anomaly"] = raw_labels == -1
        df["anomaly_score"] = scores

        # Z-score alert on p95 specifically (explainable single-metric)
        if "z_p95_ms" in df.columns:
            df["zscore_p95"] = df["z_p95_ms"].abs()
            df["zscore_alert"] = df["zscore_p95"] > self.zscore_threshold
        else:
            df["zscore_p95"] = 0.0
            df["zscore_alert"] = False

        df["severity"] = df.apply(self._classify_severity, axis=1)
        return df

    def fit_predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit on the same data and predict.

        Note: In production, fit on baseline data and predict on new runs.
        fit_predict is useful for exploratory analysis.
        Use fit_predict_baseline_split() to avoid training on the same
        windows being scored.
        """
        return self.fit(feature_df).predict(feature_df)

    def fit_predict_baseline_split(
        self,
        feature_df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        baseline_fraction: float = 0.3,
    ) -> pd.DataFrame:
        """Fit on an early chronological slice (trusted baseline) and score
        the full set against it.

        This is the production-recommended entry point: fit_predict() fits
        and scores the same window, which lets injected/real anomalies
        calibrate the model's own notion of "normal". Here the model only
        ever learns from the earliest `baseline_fraction` of the data, so
        later anomalies can't leak into the boundary that's used to catch
        them.

        Args:
            feature_df:        Feature matrix from build_perf_feature_matrix().
            timestamp_col:      Column used to order chronologically.
            baseline_fraction:  Fraction of the earliest rows used as baseline.

        Returns:
            Same output as predict(), scored for the full feature_df.
        """
        baseline_df, _ = split_baseline_period(
            feature_df, baseline_fraction=baseline_fraction, timestamp_col=timestamp_col
        )
        if len(baseline_df) < 5:
            raise ValueError(
                f"Baseline slice too small ({len(baseline_df)} rows) to fit reliably. "
                f"Increase baseline_fraction or provide more history."
            )
        return self.fit(baseline_df).predict(feature_df)

    def save(self, path: str) -> None:
        """Persist fitted detector to disk."""
        self._check_fitted()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Anomaly detector saved → {path}")

    @classmethod
    def load(cls, path: str) -> "PerformanceAnomalyDetector":
        """Load a previously fitted detector."""
        return joblib.load(path)

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Detector not fitted. Call .fit() first.")

    def _classify_severity(self, row: pd.Series) -> str:
        """Classify anomaly severity based on combined signals."""
        if not row.get("is_anomaly", False) and not row.get("zscore_alert", False):
            return "normal"
        # Critical: both Isolation Forest AND Z-score flag it
        if row.get("is_anomaly", False) and row.get("zscore_alert", False):
            return "critical"
        # Warning: only one signal fires
        return "warning"

    def summary(self, results_df: pd.DataFrame) -> dict:
        """Print and return a summary of anomaly detection results."""
        total = len(results_df)
        anomalies = results_df["is_anomaly"].sum()
        critical = (results_df["severity"] == "critical").sum()
        warnings = (results_df["severity"] == "warning").sum()

        summary = {
            "total_windows": total,
            "anomaly_count": int(anomalies),
            "anomaly_rate": round(anomalies / total, 4) if total > 0 else 0.0,
            "critical_count": int(critical),
            "warning_count": int(warnings),
            "most_anomalous_endpoint": self._most_anomalous(results_df),
        }

        print(f"\n{'='*50}")
        print("  Performance Anomaly Detection — Summary")
        print(f"{'='*50}")
        for k, v in summary.items():
            print(f"  {k:<35} {v}")

        return summary

    @staticmethod
    def _most_anomalous(df: pd.DataFrame) -> str:
        """Return the endpoint with the most anomalies."""
        if "endpoint" not in df.columns or df.empty:
            return "N/A"
        anomalies = df[df["is_anomaly"]]
        if anomalies.empty:
            return "none"
        return anomalies.groupby("endpoint").size().idxmax()
