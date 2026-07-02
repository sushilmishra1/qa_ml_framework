"""
generate_sample_data.py
-----------------------
Generates realistic synthetic test data for local development and CI dry-runs,
so the pipeline can be exercised without a real CI/test-history backfill.

Produces:
  - data/raw/junit/       : 60 JUnit XML files (one per simulated CI run)
  - data/raw/locust/      : Locust stats_history CSV with injected anomalies

No real CI system or load testing tool required.
"""

import os
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Test suite definition ─────────────────────────────────────────────────

MODULES = [
    "auth", "payments", "orders", "users", "inventory",
    "search", "notifications", "reporting", "api_gateway", "cart"
]

TEST_TEMPLATES = [
    "test_{module}_happy_path",
    "test_{module}_invalid_input",
    "test_{module}_timeout",
    "test_{module}_concurrent_access",
    "test_{module}_boundary_values",
    "test_{module}_integration",
    "test_{module}_performance",
    "test_{module}_security",
]

# Tests with intentionally high failure rates (simulating unstable areas)
FLAKY_TESTS = {
    "auth::test_auth_concurrent_access": 0.40,
    "payments::test_payments_timeout": 0.35,
    "search::test_search_performance": 0.25,
    "orders::test_orders_integration": 0.20,
}

NORMAL_FAILURE_RATE = 0.04  # ~4% baseline failure rate

# ── JUnit XML generation ──────────────────────────────────────────────────

def _random_sha() -> str:
    return "".join(random.choices(string.hexdigits[:16], k=8))


def _generate_test_suite(run_date: datetime, commit_sha: str) -> list:
    """Generate a list of (classname, testname, status, duration_ms) tuples."""
    tests = []
    for module in MODULES:
        for template in TEST_TEMPLATES:
            name = template.format(module=module)
            test_id = f"{module}::{name}"
            base_duration = random.gauss(850, 320)

            # Lookup failure probability
            failure_rate = FLAKY_TESTS.get(test_id, NORMAL_FAILURE_RATE)
            failed = random.random() < failure_rate

            # Occasional slow test (infrastructure noise)
            if random.random() < 0.03:
                base_duration *= random.uniform(3, 8)

            status = "failed" if failed else "passed"
            if random.random() < 0.005:
                status = "skipped"

            tests.append({
                "classname": module,
                "name": name,
                "status": status,
                "duration_s": max(0.01, base_duration / 1000),
                "failure_message": f"AssertionError: expected True but got False in {name}"
                                   if failed else "",
            })
    return tests


def _build_junit_xml(tests: list, run_date: datetime, commit_sha: str) -> str:
    """Build a JUnit XML string from a list of test dicts."""
    ts_root = Element("testsuites")
    suite = SubElement(ts_root, "testsuite",
                       name="qa_suite",
                       timestamp=run_date.strftime("%Y-%m-%dT%H:%M:%S"),
                       tests=str(len(tests)),
                       failures=str(sum(1 for t in tests if t["status"] == "failed")),
                       errors="0",
                       time=str(round(sum(t["duration_s"] for t in tests), 2)))

    for t in tests:
        tc = SubElement(suite, "testcase",
                        classname=t["classname"],
                        name=t["name"],
                        time=str(round(t["duration_s"], 3)))
        if t["status"] == "failed":
            fail = SubElement(tc, "failure",
                              type="AssertionError",
                              message=t["failure_message"])
            fail.text = t["failure_message"]
        elif t["status"] == "skipped":
            SubElement(tc, "skipped")

    rough = tostring(ts_root, encoding="unicode")
    reparsed = minidom.parseString(rough)
    return reparsed.toprettyxml(indent="  ")


def generate_junit_history(
    output_dir: str = "data/raw/junit",
    n_runs: int = 60,
    days_back: int = 90,
) -> None:
    """Generate N JUnit XML files spread over the last `days_back` days."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_date = datetime.now() - timedelta(days=days_back)
    interval = timedelta(days=days_back) / n_runs

    for i in range(n_runs):
        run_date = base_date + interval * i + timedelta(hours=random.randint(0, 8))
        commit_sha = _random_sha()
        tests = _generate_test_suite(run_date, commit_sha)
        xml_content = _build_junit_xml(tests, run_date, commit_sha)

        filename = f"results_{commit_sha}.xml"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(xml_content)

    print(f"Generated {n_runs} JUnit XML files → {output_dir}")


# ── Locust CSV generation ─────────────────────────────────────────────────

ENDPOINTS = [
    "POST /api/auth/login",
    "GET /api/products",
    "POST /api/orders",
    "GET /api/users/profile",
    "GET /api/search",
]


def _baseline_response_time(endpoint: str) -> tuple:
    """(median_ms, stddev) baseline per endpoint."""
    baselines = {
        "POST /api/auth/login":  (120, 20),
        "GET /api/products":     (85, 15),
        "POST /api/orders":      (230, 40),
        "GET /api/users/profile":(95, 12),
        "GET /api/search":       (180, 35),
    }
    return baselines.get(endpoint, (150, 25))


def generate_locust_history(
    output_dir: str = "data/raw/locust",
    n_windows: int = 200,
    inject_anomalies: bool = True,
) -> None:
    """Generate a synthetic Locust stats_history CSV with injected anomalies."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    base_time = datetime.now() - timedelta(hours=n_windows // 6)
    rows = []

    # Anomaly windows: 20% into the run and 70%
    anomaly_windows = set()
    if inject_anomalies:
        anomaly_start_1 = int(n_windows * 0.20)
        anomaly_start_2 = int(n_windows * 0.70)
        anomaly_windows = (
            set(range(anomaly_start_1, anomaly_start_1 + 8)) |
            set(range(anomaly_start_2, anomaly_start_2 + 5))
        )

    for i in range(n_windows):
        ts = base_time + timedelta(seconds=i * 10)
        is_anomaly = i in anomaly_windows

        for endpoint in ENDPOINTS:
            median, stddev = _baseline_response_time(endpoint)

            if is_anomaly:
                # Spike: 3–6x normal response time + elevated error rate
                spike = random.uniform(3.0, 6.0)
                median_val = median * spike + random.gauss(0, stddev * 2)
                error_rate = random.uniform(0.05, 0.25)
                rps = random.uniform(5, 15)   # throughput drops during a spike
            else:
                median_val = median + random.gauss(0, stddev)
                error_rate = random.uniform(0.0, 0.02)
                rps = random.uniform(40, 80)

            median_val = max(10, median_val)
            p95_val = median_val * random.uniform(1.5, 2.5)
            p99_val = p95_val * random.uniform(1.2, 1.8)

            rows.append({
                "Timestamp": int(ts.timestamp()),
                "User count": random.randint(50, 200),
                "Type": "GET" if endpoint.startswith("GET") else "POST",
                "Name": endpoint,
                "Requests/s": round(rps, 2),
                "Failures/s": round(rps * error_rate, 3),
                "50%": round(median_val, 1),
                "66%": round(median_val * 1.2, 1),
                "75%": round(median_val * 1.4, 1),
                "80%": round(median_val * 1.5, 1),
                "90%": round(median_val * 1.8, 1),
                "95%": round(p95_val, 1),
                "98%": round(p95_val * 1.3, 1),
                "99%": round(p99_val, 1),
                "99.9%": round(p99_val * 1.5, 1),
                "99.99%": round(p99_val * 2.0, 1),
                "100%": round(p99_val * 3.0, 1),
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "loadtest_stats_history.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated Locust history CSV ({len(df)} rows) → {out_path}")
    print(f"  Anomaly windows injected at positions: "
          f"{sorted(anomaly_windows)[:5]}... ({len(anomaly_windows)} total)")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Console output below uses unicode arrows; Windows terminals default to
    # cp1252, which raises UnicodeEncodeError on those characters mid-run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Generating synthetic QA data...\n")
    generate_junit_history()
    generate_locust_history()
    print("\nDone. Run: python scripts/run_pipeline.py")
