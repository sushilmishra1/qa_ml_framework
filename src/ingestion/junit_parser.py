"""
junit_parser.py
---------------
Parses JUnit XML reports (pytest, TestNG, Selenium) into a flat pandas DataFrame.

Industry standard: pytest --junitxml=results.xml produces this format.
Every CI system (GitHub Actions, Jenkins, TeamCity) consumes and produces it.

DataFrame schema produced:
    test_id        : str   - unique test identifier (classname::name)
    test_name      : str   - short test name
    classname      : str   - module/class path
    status         : str   - 'passed' | 'failed' | 'error' | 'skipped'
    duration_ms    : float - execution time in milliseconds
    run_timestamp  : datetime - when the test ran
    commit_sha     : str   - git commit hash (from XML attribute or filename)
    failure_message: str   - failure message if failed, else ''
    error_type     : str   - exception type if failed, else ''
"""

import os
import glob
from datetime import datetime
from xml.etree import ElementTree as ET

import pandas as pd


def parse_junit_file(xml_path: str, commit_sha: str = "unknown") -> pd.DataFrame:
    """Parse a single JUnit XML file into a DataFrame.

    Args:
        xml_path: Path to the JUnit XML report file.
        commit_sha: Git commit SHA to tag these results with.

    Returns:
        DataFrame with columns matching the schema above.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Handle both <testsuites> root and <testsuite> root
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    elif root.tag == "testsuite":
        suites = [root]
    else:
        raise ValueError(f"Unexpected root tag: {root.tag} in {xml_path}")

    # Extract timestamp: prefer the first <testsuite> child's attribute
    ts_source = suites[0] if suites else root
    run_timestamp = _parse_timestamp(ts_source.get("timestamp"))

    rows = []
    for suite in suites:
        for tc in suite.findall("testcase"):
            name = tc.get("name", "")
            classname = tc.get("classname", "")
            test_id = f"{classname}::{name}" if classname else name
            duration_ms = float(tc.get("time", 0)) * 1000  # convert seconds → ms

            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if failure is not None:
                status = "failed"
                failure_message = failure.get("message", failure.text or "")
                error_type = failure.get("type", "")
            elif error is not None:
                status = "error"
                failure_message = error.get("message", error.text or "")
                error_type = error.get("type", "")
            elif skipped is not None:
                status = "skipped"
                failure_message = ""
                error_type = ""
            else:
                status = "passed"
                failure_message = ""
                error_type = ""

            rows.append({
                "test_id": test_id,
                "test_name": name,
                "classname": classname,
                "status": status,
                "duration_ms": duration_ms,
                "run_timestamp": run_timestamp,
                "commit_sha": commit_sha,
                "failure_message": failure_message,
                "error_type": error_type,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
    return df


def parse_junit_directory(dir_path: str) -> pd.DataFrame:
    """Parse all JUnit XML files in a directory recursively.

    Filenames are expected to optionally encode the commit SHA:
        results_abc1234.xml  →  commit_sha = 'abc1234'
        results.xml          →  commit_sha = 'unknown'

    Args:
        dir_path: Directory containing JUnit XML reports.

    Returns:
        Combined DataFrame of all test runs, sorted by run_timestamp.
    """
    xml_files = glob.glob(os.path.join(dir_path, "**", "*.xml"), recursive=True)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {dir_path}")

    frames = []
    for xml_path in sorted(xml_files):
        basename = os.path.splitext(os.path.basename(xml_path))[0]
        # Convention: filename may end with _<sha> e.g. results_abc1234
        parts = basename.rsplit("_", 1)
        commit_sha = parts[-1] if len(parts) == 2 and len(parts[-1]) >= 7 else "unknown"

        try:
            df = parse_junit_file(xml_path, commit_sha=commit_sha)
            frames.append(df)
        except Exception as exc:
            print(f"[WARNING] Skipping {xml_path}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("run_timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO or partial timestamp strings from JUnit XML."""
    if not ts_str:
        return datetime.now()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return datetime.now()
