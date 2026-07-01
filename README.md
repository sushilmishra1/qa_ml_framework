# qa-ml-framework

> **ML-powered QA intelligence** — test failure prediction, risk-based prioritisation, and performance anomaly detection.  
> Pure Python · No AI/LLM dependencies · Interview-ready · GitHub Actions CI gate included.

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
│   └── exploration.ipynb       # Demo notebook for interviews
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

Output: flagged time windows with anomaly scores, suitable for CI gate or Grafana alerting.

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

## Interview talking points

- **Why Random Forest over XGBoost?** Interpretability — `feature_importances_` lets you explain to a QA lead *why* a test was flagged.
- **Why time-split CV?** Test runs are time-ordered. Random k-fold leaks future data into training.
- **Why Isolation Forest for flakiness?** No labelled dataset required — unsupervised detection of anomalous pass/fail patterns.
- **Why not accuracy?** Class imbalance — 95% pass rate means a trivial classifier gets 95% accuracy. We optimise F1 and precision@k.
- **Performance anomaly without LLM** — pure statistical approach: Z-score for simple thresholds, Isolation Forest for multivariate anomalies across response_time + throughput + error_rate simultaneously.

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
