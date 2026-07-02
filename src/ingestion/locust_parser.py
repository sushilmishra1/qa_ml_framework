"""
locust_parser.py
----------------
Parses Locust load test CSV exports into a pandas DataFrame for anomaly detection.

Locust generates two CSV files after a run:
  - *_stats.csv         : aggregate stats per endpoint
  - *_stats_history.csv : time-series metrics (one row per 10s window)

We use stats_history for anomaly detection (time-series approach).

DataFrame schema (from stats_history):
    timestamp       : datetime - window start
    endpoint        : str      - request name / URL
    requests        : int      - requests in this window
    failures        : int      - failed requests
    median_ms       : float    - median response time (ms)
    p95_ms          : float    - 95th percentile response time (ms)
    p99_ms          : float    - 99th percentile response time (ms)
    avg_ms          : float    - mean response time (ms)
    min_ms          : float    - minimum response time (ms)
    max_ms          : float    - maximum response time (ms)
    rps             : float    - requests per second
    error_rate      : float    - failure rate 0.0-1.0
"""

import os
import glob

import pandas as pd


# Locust CSV column names vary slightly between versions
# These are the canonical names for Locust ≥ 2.x
_HISTORY_COLUMNS = {
    "Timestamp": "timestamp",
    "User count": "user_count",
    "Type": "request_type",
    "Name": "endpoint",
    "Requests/s": "rps",
    "Failures/s": "failures_per_s",
    "50%": "median_ms",
    "66%": "p66_ms",
    "75%": "p75_ms",
    "80%": "p80_ms",
    "90%": "p90_ms",
    "95%": "p95_ms",
    "98%": "p98_ms",
    "99%": "p99_ms",
    "99.9%": "p999_ms",
    "99.99%": "p9999_ms",
    "100%": "max_ms",
}

_STATS_COLUMNS = {
    "Name": "endpoint",
    "# Requests": "requests",
    "# Fails": "failures",
    "Median (ms)": "median_ms",
    "95%ile (ms)": "p95_ms",
    "99%ile (ms)": "p99_ms",
    "Average (ms)": "avg_ms",
    "Min (ms)": "min_ms",
    "Max (ms)": "max_ms",
    "Average size (bytes)": "avg_size_bytes",
    "Current RPS": "rps",
    "Current Failures/s": "failures_per_s",
}


def parse_locust_history(csv_path: str) -> pd.DataFrame:
    """Parse a Locust *_stats_history.csv into a clean DataFrame.

    Args:
        csv_path: Path to the Locust stats history CSV.

    Returns:
        DataFrame with time-series performance metrics per endpoint.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={k: v for k, v in _HISTORY_COLUMNS.items() if k in df.columns},
              inplace=True)

    # Convert Unix timestamp to datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")

    # Derive error_rate from failures_per_s / rps (avoid div-by-zero)
    if "failures_per_s" in df.columns and "rps" in df.columns:
        df["error_rate"] = df.apply(
            lambda r: r["failures_per_s"] / r["rps"] if r["rps"] > 0 else 0.0,
            axis=1
        ).clip(0.0, 1.0)

    # Filter out the aggregate row (endpoint == "Aggregated")
    if "endpoint" in df.columns:
        df = df[df["endpoint"] != "Aggregated"].copy()

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def parse_locust_stats(csv_path: str) -> pd.DataFrame:
    """Parse a Locust *_stats.csv (aggregate summary) into a DataFrame.

    Args:
        csv_path: Path to the Locust stats CSV.

    Returns:
        DataFrame with one row per endpoint - aggregate for the full run.
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={k: v for k, v in _STATS_COLUMNS.items() if k in df.columns},
              inplace=True)

    if "endpoint" in df.columns:
        df = df[df["endpoint"] != "Aggregated"].copy()

    # Derive error_rate
    if "failures" in df.columns and "requests" in df.columns:
        df["error_rate"] = df.apply(
            lambda r: r["failures"] / r["requests"] if r["requests"] > 0 else 0.0,
            axis=1
        ).clip(0.0, 1.0)

    df.reset_index(drop=True, inplace=True)
    return df


def parse_locust_directory(dir_path: str) -> dict:
    """Find and parse all Locust CSV files in a directory.

    Returns:
        dict with keys 'history' and 'stats', each a combined DataFrame.
    """
    history_files = glob.glob(os.path.join(dir_path, "**", "*_stats_history.csv"),
                              recursive=True)
    stats_files = glob.glob(os.path.join(dir_path, "**", "*_stats.csv"),
                            recursive=True)
    # Exclude history files from stats list
    stats_files = [f for f in stats_files if "_stats_history" not in f]

    history_frames = [parse_locust_history(f) for f in history_files]
    stats_frames = [parse_locust_stats(f) for f in stats_files]

    return {
        "history": pd.concat(history_frames, ignore_index=True) if history_frames
                   else pd.DataFrame(),
        "stats": pd.concat(stats_frames, ignore_index=True) if stats_frames
                 else pd.DataFrame(),
    }
