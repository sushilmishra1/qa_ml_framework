"""
run_pipeline.py
---------------
End-to-end pipeline: ingest -> features -> train -> predict -> anomaly detect -> report.

Run this after generate_sample_data.py:
    python scripts/run_pipeline.py

For CI use (after training):
    python scripts/run_pipeline.py --ci-mode --predictions-only
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from src.ingestion.junit_parser import parse_junit_directory
from src.ingestion.locust_parser import parse_locust_history
from src.features.test_features import build_feature_matrix
from src.features.perf_features import build_perf_feature_matrix
from src.models.failure_predictor import FailurePredictor
from src.perf_anomaly.anomaly_detector import PerformanceAnomalyDetector
from src.ci_gate.gate import run_gate
from src.ci_gate.report import generate_html_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_full_pipeline(config_path: str = "config.yaml", ci_mode: bool = False):
    """Run the complete ml-anomaly-detection-framework pipeline."""
    config = load_config(config_path)
    print("\n" + "="*60)
    print("  ml-anomaly-detection-framework - Full Pipeline")
    print("="*60)

    # --- Step 1: Ingest test history ---
    print("\n[1/6] Ingesting JUnit XML history...")
    junit_dir = os.path.join(config["data"]["raw_dir"], "junit")
    if not os.path.exists(junit_dir):
        print(f"  ERROR: {junit_dir} not found. Run: python scripts/generate_sample_data.py")
        sys.exit(1)

    history_df = parse_junit_directory(junit_dir)
    print(f"  Loaded {len(history_df)} test results across "
          f"{history_df['test_id'].nunique()} unique tests")

    # --- Step 2: Feature engineering ---
    print("\n[2/6] Building feature matrix...")
    rolling_days = config["features"]["rolling_window_days"]
    feature_df = build_feature_matrix(history_df, rolling_days=rolling_days)
    print(f"  Built {len(feature_df)} feature rows "
          f"(positive rate: {feature_df['is_failed'].mean():.3f})")

    # Save features
    processed_dir = config["data"]["processed_dir"]
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(os.path.join(processed_dir, "features.parquet"), index=False)

    # --- Step 3: Train failure predictor ---
    print("\n[3/6] Training failure predictor (Random Forest)...")
    predictor = FailurePredictor(
        model_type="random_forest",
        n_estimators=config["model"]["n_estimators"],
        random_state=config["model"]["random_state"],
    )
    metrics = predictor.train(
        feature_df,
        split_days=config["model"]["test_split_days"],
    )
    models_dir = "models"
    Path(models_dir).mkdir(exist_ok=True)
    predictor.save(os.path.join(models_dir, "failure_predictor.pkl"))

    # --- Step 4: Generate predictions for current tests ---
    print("\n[4/6] Generating predictions...")
    # Use the most recent snapshot of each test (simulate "current run")
    latest = (
        feature_df
        .sort_values("run_timestamp")
        .groupby("test_id")
        .last()
        .reset_index()
    )
    predictions = predictor.predict(latest)
    predictions_path = os.path.join(config["reports"]["output_dir"], "predictions.csv")
    Path(predictions_path).parent.mkdir(exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    print(f"  Predicted {len(predictions)} tests -> {predictions_path}")
    print(f"  High-risk (>0.35): {(predictions['p_fail'] >= 0.35).sum()}")

    # --- Step 5: Performance anomaly detection ---
    print("\n[5/6] Running performance anomaly detection...")
    anomaly_results = None
    locust_dir = os.path.join(config["data"]["raw_dir"], "locust")
    locust_csv = os.path.join(locust_dir, "loadtest_stats_history.csv")

    if os.path.exists(locust_csv):
        from src.ingestion.locust_parser import parse_locust_history
        locust_df = parse_locust_history(locust_csv)
        perf_features = build_perf_feature_matrix(
            locust_df,
            baseline_window=config["anomaly"]["baseline_window"],
        )
        detector = PerformanceAnomalyDetector(
            contamination=config["anomaly"]["contamination"],
            zscore_threshold=config["anomaly"]["zscore_threshold"],
        )
        anomaly_results = detector.fit_predict_baseline_split(
            perf_features,
            baseline_fraction=config["anomaly"].get("baseline_fraction", 0.3),
        )
        summary = detector.summary(anomaly_results)
        detector.save(os.path.join(models_dir, "anomaly_detector.pkl"))
    else:
        print(f"  Skipping perf anomaly (no Locust CSV at {locust_csv})")

    # --- Step 6: CI gate + HTML report ---
    print("\n[6/6] Running CI gate and generating report...")
    gate_result_path = os.path.join(config["reports"]["output_dir"], "gate_result.json")
    exit_code = run_gate(
        predictions_path=predictions_path,
        config_path=config_path,
        output_path=gate_result_path,
    )

    import json
    with open(gate_result_path) as f:
        gate_result = json.load(f)

    report_path = generate_html_report(
        predictions_df=predictions,
        gate_result=gate_result,
        feature_importances=predictor.feature_importances_,
        anomaly_df=anomaly_results,
        output_path=os.path.join(config["reports"]["output_dir"], "prediction_report.html"),
    )

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Report:     {report_path}")
    print(f"  Gate:       {'PASSED' if gate_result['gate_passed'] else 'FAILED'}")
    print(f"{'='*60}\n")

    if ci_mode:
        sys.exit(exit_code)


if __name__ == "__main__":
    # Console output uses unicode arrows/checkmarks; Windows terminals default
    # to cp1252, which raises UnicodeEncodeError on those characters mid-run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="ml-anomaly-detection-framework - Pipeline Runner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ci-mode", action="store_true",
                        help="Exit with gate exit code (for CI integration)")
    args = parser.parse_args()
    run_full_pipeline(config_path=args.config, ci_mode=args.ci_mode)
