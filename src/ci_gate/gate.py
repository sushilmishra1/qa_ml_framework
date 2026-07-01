"""
gate.py
-------
CI gate: reads model predictions, applies thresholds, exits 0 (pass) or 1 (fail).

This is the script called by GitHub Actions / Jenkins as the last pipeline step.
It follows the same pattern as code coverage gates — set a floor, block regressions.

Exit codes (standard CI convention):
    0 = gate passed (safe to merge)
    1 = gate failed (block PR)
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_gate(
    predictions_path: str,
    config_path: str = "config.yaml",
    output_path: str = "reports/gate_result.json",
    top_n: int = None,
    threshold: float = None,
) -> int:
    """Run the CI gate against a predictions CSV.

    Args:
        predictions_path: Path to CSV with columns: test_id, p_fail
        config_path:      Path to config.yaml
        output_path:      Where to write the JSON gate result
        top_n:            Override config top_n_tests
        threshold:        Override config risk_threshold

    Returns:
        0 if gate passes, 1 if gate fails.
    """
    config = load_config(config_path)
    gate_cfg = config.get("ci_gate", {})

    risk_threshold = threshold or gate_cfg.get("risk_threshold", 0.35)
    top_n_tests = top_n or gate_cfg.get("top_n_tests", 200)

    # Load predictions
    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"[GATE ERROR] Predictions file not found: {predictions_path}")
        return 1

    if "p_fail" not in df.columns:
        print("[GATE ERROR] Predictions CSV must contain a 'p_fail' column.")
        return 1

    df = df.sort_values("p_fail", ascending=False).reset_index(drop=True)

    # Compute the mean risk score of the top-N highest-risk tests
    top_tests = df.head(top_n_tests)
    mean_top_risk = top_tests["p_fail"].mean()

    # Individual tests that exceed the threshold
    high_risk = df[df["p_fail"] >= risk_threshold]

    gate_passed = mean_top_risk < risk_threshold

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "gate_passed": bool(gate_passed),
        "exit_code": 0 if gate_passed else 1,
        "mean_top_risk": round(float(mean_top_risk), 4),
        "risk_threshold": risk_threshold,
        "top_n_evaluated": top_n_tests,
        "total_tests": len(df),
        "high_risk_count": len(high_risk),
        "top_10_tests": top_tests.head(10)[["test_id", "p_fail"]].to_dict("records"),
    }

    # Write JSON report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print CI-friendly summary
    status_icon = "✅" if gate_passed else "❌"
    print(f"\n{'='*55}")
    print(f"  {status_icon}  QA ML Gate Result")
    print(f"{'='*55}")
    print(f"  Mean risk score (top-{top_n_tests} tests): {mean_top_risk:.4f}")
    print(f"  Threshold:                          {risk_threshold:.4f}")
    print(f"  High-risk tests (>= threshold):     {len(high_risk)}")
    print(f"  Gate status:                        {'PASSED' if gate_passed else 'FAILED'}")
    print(f"{'='*55}")

    if not gate_passed:
        print(f"\n  ❌ Gate FAILED — {len(high_risk)} tests exceed risk threshold {risk_threshold}")
        print("  Top high-risk tests:")
        for _, row in high_risk.head(5).iterrows():
            print(f"    [{row['p_fail']:.3f}]  {row.get('test_id', 'unknown')}")
        print(f"\n  Gate result saved → {output_path}")
        return 1

    print(f"\n  ✅ Gate PASSED — safe to merge")
    print(f"  Gate result saved → {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="QA ML CI Gate — blocks PRs if predicted failure risk is too high"
    )
    parser.add_argument(
        "predictions",
        help="Path to predictions CSV (test_id, p_fail)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--output", default="reports/gate_result.json",
        help="Path to write gate result JSON"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override risk threshold from config"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Override top_n_tests from config"
    )
    args = parser.parse_args()

    exit_code = run_gate(
        predictions_path=args.predictions,
        config_path=args.config,
        output_path=args.output,
        threshold=args.threshold,
        top_n=args.top_n,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
