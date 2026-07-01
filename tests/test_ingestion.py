"""
test_ingestion.py
-----------------
Unit tests for JUnit XML and Locust CSV parsers.
Run with: pytest tests/test_ingestion.py -v
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from src.ingestion.junit_parser import parse_junit_file, parse_junit_directory
from src.ingestion.locust_parser import parse_locust_history, parse_locust_stats


# ── Fixtures ──────────────────────────────────────────────────────────────

SAMPLE_JUNIT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="auth_tests" timestamp="2024-01-15T10:30:00" tests="4" failures="1">
    <testcase classname="auth" name="test_login_happy_path" time="0.123"/>
    <testcase classname="auth" name="test_login_invalid_creds" time="0.085">
      <failure type="AssertionError" message="Expected 401, got 200"/>
    </testcase>
    <testcase classname="auth" name="test_logout" time="0.045"/>
    <testcase classname="auth" name="test_refresh_token" time="0.200">
      <skipped/>
    </testcase>
  </testsuite>
</testsuites>"""

SAMPLE_LOCUST_HISTORY_CSV = """Timestamp,User count,Type,Name,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
1700000000,100,GET,GET /api/products,50.2,0.1,85,90,95,100,110,130,150,180,250,400,600
1700000010,100,GET,GET /api/products,48.5,0.2,90,95,100,105,115,140,160,190,260,420,650
1700000020,100,GET,GET /api/products,45.0,2.5,350,380,400,420,450,520,600,700,900,1200,2000
1700000030,100,GET,GET /api/products,51.1,0.1,88,93,98,102,112,132,152,182,252,402,602
"""


@pytest.fixture
def junit_xml_file(tmp_path):
    xml_file = tmp_path / "results_abc12345.xml"
    xml_file.write_text(SAMPLE_JUNIT_XML)
    return str(xml_file)


@pytest.fixture
def locust_csv_file(tmp_path):
    csv_file = tmp_path / "loadtest_stats_history.csv"
    csv_file.write_text(SAMPLE_LOCUST_HISTORY_CSV)
    return str(csv_file)


# ── JUnit parser tests ─────────────────────────────────────────────────────

class TestJUnitParser:

    def test_parse_returns_dataframe(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        assert isinstance(df, pd.DataFrame)

    def test_parse_correct_row_count(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        assert len(df) == 4

    def test_parse_status_values(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        assert set(df["status"].unique()).issubset(
            {"passed", "failed", "error", "skipped"}
        )

    def test_parse_failure_detected(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        failures = df[df["status"] == "failed"]
        assert len(failures) == 1
        assert "test_login_invalid_creds" in failures.iloc[0]["test_id"]

    def test_parse_failure_message_captured(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        failure_row = df[df["status"] == "failed"].iloc[0]
        assert "401" in failure_row["failure_message"]

    def test_parse_duration_in_ms(self, junit_xml_file):
        """Duration should be converted from seconds to milliseconds."""
        df = parse_junit_file(junit_xml_file)
        happy_path = df[df["test_name"] == "test_login_happy_path"].iloc[0]
        assert abs(happy_path["duration_ms"] - 123.0) < 1.0

    def test_parse_commit_sha_extracted(self, junit_xml_file):
        """Commit SHA should be extracted from filename convention."""
        df = parse_junit_file(junit_xml_file, commit_sha="abc12345")
        assert all(df["commit_sha"] == "abc12345")

    def test_parse_test_id_format(self, junit_xml_file):
        """test_id should be classname::name."""
        df = parse_junit_file(junit_xml_file)
        assert all("::" in tid for tid in df["test_id"])

    def test_parse_skipped_detected(self, junit_xml_file):
        df = parse_junit_file(junit_xml_file)
        skipped = df[df["status"] == "skipped"]
        assert len(skipped) == 1

    def test_parse_directory(self, tmp_path):
        """parse_junit_directory should concatenate multiple files."""
        for i in range(3):
            f = tmp_path / f"results_sha{i:07d}.xml"
            f.write_text(SAMPLE_JUNIT_XML)
        df = parse_junit_directory(str(tmp_path))
        assert len(df) == 12  # 3 files × 4 tests

    def test_parse_directory_sorted_by_time(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"results_sha{i:07d}.xml"
            f.write_text(SAMPLE_JUNIT_XML)
        df = parse_junit_directory(str(tmp_path))
        timestamps = df["run_timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_parse_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_junit_directory("/nonexistent/path")


# ── Locust parser tests ────────────────────────────────────────────────────

class TestLocustParser:

    def test_parse_returns_dataframe(self, locust_csv_file):
        df = parse_locust_history(locust_csv_file)
        assert isinstance(df, pd.DataFrame)

    def test_parse_row_count(self, locust_csv_file):
        df = parse_locust_history(locust_csv_file)
        assert len(df) == 4

    def test_parse_timestamp_is_datetime(self, locust_csv_file):
        df = parse_locust_history(locust_csv_file)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_parse_error_rate_derived(self, locust_csv_file):
        """error_rate should be derived from failures_per_s / rps."""
        df = parse_locust_history(locust_csv_file)
        assert "error_rate" in df.columns
        assert (df["error_rate"] >= 0.0).all()
        assert (df["error_rate"] <= 1.0).all()

    def test_parse_error_rate_spike_detected(self, locust_csv_file):
        """Third row has Failures/s=2.5 and Requests/s=45.0 → error_rate ≈ 0.055."""
        df = parse_locust_history(locust_csv_file)
        spike_row = df.iloc[2]
        assert spike_row["error_rate"] > 0.04

    def test_parse_columns_renamed(self, locust_csv_file):
        df = parse_locust_history(locust_csv_file)
        assert "median_ms" in df.columns
        assert "p95_ms" in df.columns
        assert "rps" in df.columns

    def test_parse_aggregated_row_excluded(self, tmp_path):
        """Rows with Name='Aggregated' should be filtered out."""
        csv = tmp_path / "test_stats_history.csv"
        csv.write_text(
            "Timestamp,User count,Type,Name,Requests/s,Failures/s,"
            "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%\n"
            "1700000000,100,GET,GET /api/test,50,0,85,90,95,100,110,130,150,180,250,400,600\n"
            "1700000000,100,GET,Aggregated,50,0,85,90,95,100,110,130,150,180,250,400,600\n"
        )
        df = parse_locust_history(str(csv))
        assert len(df) == 1
        assert "Aggregated" not in df["endpoint"].values
