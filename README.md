# qa-ml-framework

> **ML-powered QA intelligence** — test failure prediction, risk-based prioritisation, and performance anomaly detection.  
> Pure Python · No AI/LLM dependencies · No cloud services required · GitHub Actions CI gate included.

---

## What this framework does

| Module | Problem solved | Algorithm |
|--------|---------------|-----------|
| **Test Failure Prediction** | Which tests are most likely to fail on this commit? | Random Forest + Logistic Regression |
| **Flaky Test Detection** | Which tests pass/fail randomly without code changes? | Isolation Forest (unsupervised) |
| **Risk-Based Prioritisation** | Given 10,000 tests and 10 minutes, which 200 should run? | Ranked scoring pipeline |
| **Performance Anomaly Detection** | Did response times spike in this load test run? | Isolation Forest + Z-score |
| **CI Gate** | Block PRs if predicted failure risk is too high | Score threshold enforcement |

---

## Project structure

```
qa-ml-framework/
├── data/
│   ├── raw/                    # JUnit XML reports, Locust CSV exports
│   └── processed/              # Engineered feature DataFrames (parquet)
├── src/
│   ├── ingestion/
│   │   ├── junit_parser.py     # Parse JUnit XML → DataFrame
│   │   └── locust_parser.py    # Parse Locust CSV → DataFrame
│   ├── features/
│   │   ├── test_features.py    # Feature engineering for failure prediction
│   │   └── perf_features.py    # Feature engineering for perf anomaly
│   ├── models/
│   │   ├── failure_predictor.py     # Random Forest failure prediction
│   │   ├── flaky_detector.py        # Isolation Forest flakiness
│   │   └── model_utils.py           # Train/eval helpers, time-split CV
│   ├── perf_anomaly/
│   │   ├── anomaly_detector.py      # Isolation Forest on perf metrics
│   │   └── baseline.py              # Baseline stats (mean, stddev, percentiles)
│   └── ci_gate/
│       ├── gate.py                  # Scoring + pass/fail decision
│       └── report.py                # HTML + JSON report generation
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_perf_anomaly.py
├── notebooks/
│   └── exploration.ipynb       # Notebook for exploring predictions and feature importances
├── reports/                    # Generated HTML/JSON reports
├── scripts/
│   ├── generate_sample_data.py # Synthetic data generator (no real tool needed)
│   └── run_pipeline.py         # End-to-end pipeline runner
├── .github/
│   └── workflows/
│       ├── ci.yml              # Main CI: test + lint + gate
│       └── ml_gate.yml         # ML risk gate on PR
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/qa-ml-framework.git
cd qa-ml-framework
pip install -r requirements.txt

# 2. Generate synthetic sample data
python scripts/generate_sample_data.py

# 3. Run the full pipeline
python scripts/run_pipeline.py

# 4. View the HTML report
open reports/prediction_report.html

# 5. Run framework tests
pytest tests/ -v --junitxml=reports/test_results.xml
```

---

## Data sources (industry standard)

### Test history — JUnit XML
Every major test runner outputs JUnit XML:
- **pytest**: `pytest --junitxml=results.xml`
- **TestNG / JUnit**: default output
- **Selenium Grid**: via `pytest-selenium`
- **Cypress / Playwright**: via JUnit reporter plugins

### Performance data — Locust CSV
Locust exports `*_stats.csv` and `*_stats_history.csv` after every run.  
JMeter exports similar CSV via the Summary Report listener.

---

## Feature engineering — what the model sees

| Feature | Source | Why it matters |
|---------|--------|---------------|
| `failure_rate_30d` | JUnit XML history | Rolling 30-day failure rate — most predictive |
| `avg_duration_ms` | JUnit XML | Slow tests = infrastructure dependency = flakiness proxy |
| `duration_stddev` | JUnit XML | High variance = flaky behaviour |
| `days_since_last_fail` | JUnit XML | Staleness signal — recent failures more relevant |
| `consec_alternating` | JUnit XML | Pass→fail→pass on same SHA = flaky flag |
| `file_overlap_score` | git diff + coverage | How many changed files overlap test coverage |
| `module_churn_7d` | git log | Code churn in the module under test |
| `author_fail_rate` | JUnit + git | Author-level historical defect rate |

---

## Performance anomaly detection

The `perf_anomaly` module analyses Locust/JMeter output for:
- Response time spikes (Isolation Forest)
- Error rate surges (Z-score threshold)
- Throughput drops (rolling baseline comparison)
- P95/P99 latency regressions

Output: flagged time windows with anomaly scores, rendered in the "Performance Anomalies"
section of `reports/prediction_report.html`, suitable for CI gate or Grafana alerting.

**Fitting strategy**: the pipeline uses `fit_predict_baseline_split()`, which fits the
Isolation Forest only on the earliest `anomaly.baseline_fraction` of the run (a trusted
baseline period) and scores the rest against it. This avoids the leakage of the simpler
`fit_predict()` convenience method, which fits and scores the same window — letting real
anomalies calibrate the detector's own notion of "normal."

---

## CI gate behaviour

The gate reads a `risk_threshold` from `config.yaml`. For each test in the current suite:

```
risk_score = model.predict_proba(features)[1]   # P(fail)

if mean(top_10_risk_scores) > threshold:
    exit(1)   # Block the PR
else:
    exit(0)   # Allow merge
```

---

## Design decisions

- **Random Forest over XGBoost**: `feature_importances_` gives a QA lead a direct answer for *why* a test was flagged, which matters when the model is going to block a PR — an unexplainable rejection erodes trust in the gate fast.
- **Time-split CV, not k-fold**: test runs are time-ordered. Random k-fold would let future runs leak into training and inflate offline metrics beyond what the model can actually do in production.
- **Isolation Forest for flakiness**: there's no labelled "this test is flaky" dataset to supervise against, so detection has to be unsupervised.
- **F1/precision@k over accuracy**: the pass rate is ~95%, so a trivial "always predict pass" classifier would already score 95% accuracy while catching zero real failures. Accuracy would hide exactly the failure mode this tool exists to catch.
- **Statistical perf anomaly detection, no LLM**: Z-score for simple single-metric thresholds, Isolation Forest for multivariate anomalies across response_time + throughput + error_rate together — cheaper, faster, and auditable compared to a model-based approach for a problem that doesn't need one.

---

## Tech stack

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data ingestion, feature engineering |
| scikit-learn | ≥1.4 | Random Forest, Logistic Regression, Isolation Forest |
| numpy | ≥1.26 | Numerical operations |
| matplotlib | ≥3.8 | Report charts |
| pytest | ≥8.0 | Framework tests |
| PyYAML | ≥6.0 | Config management |
| lxml | ≥5.0 | JUnit XML parsing |

**No AI/LLM dependencies. No cloud services required.**
