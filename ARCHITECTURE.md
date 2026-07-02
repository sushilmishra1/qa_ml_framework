# QA ML Framework — Architecture

A visual walkthrough of how raw test/perf data becomes a merge-blocking risk score. Diagrams render natively on GitHub (Mermaid).

---

## 1. System overview

```mermaid
flowchart TB
    subgraph SRC["Data Sources (industry-standard formats)"]
        A1["JUnit XML\npytest / TestNG / Selenium / Cypress / Playwright"]
        A2["Locust / JMeter CSV\n*_stats.csv, *_stats_history.csv"]
        A3["git log / git diff\nchurn, file overlap, author history"]
    end

    subgraph ING["Ingestion — src/ingestion/"]
        B1["junit_parser.py\nXML -> DataFrame"]
        B2["locust_parser.py\nCSV -> DataFrame"]
    end

    subgraph FEAT["Feature Engineering — src/features/"]
        C1["test_features.py\nfailure_rate_30d, duration_stddev,\nconsec_alternating, churn, author_fail_rate"]
        C2["perf_features.py\nrolling baseline mean/stddev/percentiles"]
    end

    subgraph STORE["Storage"]
        D1[("data/processed/\n*.parquet")]
    end

    subgraph MODEL["ML Models — src/models/ & src/perf_anomaly/"]
        E1["failure_predictor.py\nRandom Forest + Logistic Regression\ntime-split CV"]
        E2["flaky_detector.py\nIsolation Forest (unsupervised)"]
        E3["anomaly_detector.py\nIsolation Forest + Z-score"]
    end

    subgraph PERSIST["Model Persistence"]
        F1[("models/*.pkl")]
    end

    subgraph GATE["CI Gate — src/ci_gate/"]
        G1["gate.py\nscore top-N tests vs risk_threshold\npass / fail decision"]
        G2["report.py\nHTML + JSON report"]
    end

    subgraph OUT["Outputs"]
        H1["reports/prediction_report.html"]
        H2["reports/gate_result.json"]
        H3["GitHub PR comment\n(pass/fail + risk table)"]
    end

    A1 --> B1 --> C1
    A2 --> B2 --> C2
    A3 --> C1
    C1 --> D1
    C2 --> D1
    D1 --> E1
    D1 --> E2
    D1 --> E3
    E1 --> F1
    E3 --> F1
    E1 --> G1
    E3 --> G1
    G1 --> G2
    G2 --> H1
    G1 --> H2
    H2 --> H3
```

---

## 2. Pipeline execution order (`scripts/run_pipeline.py`)

```mermaid
sequenceDiagram
    participant CI as GitHub Actions
    participant Pipe as run_pipeline.py
    participant Ing as ingestion
    participant Feat as features
    participant Model as failure_predictor
    participant Anom as anomaly_detector
    participant Gate as ci_gate

    CI->>Pipe: python scripts/run_pipeline.py --ci-mode
    Pipe->>Ing: [1/6] parse_junit_directory()
    Ing-->>Pipe: history_df
    Pipe->>Feat: [2/6] build_feature_matrix()
    Feat-->>Pipe: feature_df -> data/processed/features.parquet
    Pipe->>Model: [3/6] train(feature_df, time-split CV)
    Model-->>Pipe: metrics, model.pkl
    Pipe->>Model: [4/6] predict(latest snapshot per test)
    Model-->>Pipe: predictions.csv (p_fail per test)
    Pipe->>Anom: [5/6] fit_predict(perf_features)
    Anom-->>Pipe: anomaly_results
    Pipe->>Gate: [6/6] run_gate(predictions, config.risk_threshold)
    Gate-->>Pipe: gate_result.json (pass/fail)
    Pipe->>CI: exit(0) merge OK / exit(1) block PR
```

---

## 3. CI/CD integration (`.github/workflows/ci.yml`)

```mermaid
flowchart LR
    PR["PR opened / push"] --> T["Job: test\n(Python 3.10/3.11/3.12 matrix)\npytest + coverage >=70%"]
    T --> MG{"pull_request event?"}
    MG -- yes --> G1["Job: ml-gate\ngenerate/ingest data -> run_pipeline.py"]
    G1 --> G2["ci_gate.gate\nmean(top-10 risk) vs threshold"]
    G2 -- pass --> M["✅ Merge allowed"]
    G2 -- fail --> B["❌ PR blocked"]
    G2 --> Art["Upload artifacts:\nprediction_report.html, gate_result.json"]
    G2 --> Cmt["Auto-comment on PR\nrisk table + pass/fail"]
```

---

## 4. Component responsibility map

| Layer | Module | Responsibility | Technique |
|---|---|---|---|
| Ingestion | `src/ingestion/junit_parser.py` | XML → tabular test history | lxml |
| Ingestion | `src/ingestion/locust_parser.py` | Perf CSV → tabular time series | pandas |
| Features | `src/features/test_features.py` | Rolling failure rate, flakiness signals, churn | pandas |
| Features | `src/features/perf_features.py` | Rolling baseline stats | pandas |
| Model | `src/models/failure_predictor.py` | P(test fails on this commit) | Random Forest + Logistic Regression, time-split CV |
| Model | `src/models/flaky_detector.py` | Flag non-deterministic tests | Isolation Forest (unsupervised) |
| Model | `src/perf_anomaly/anomaly_detector.py` | Perf regression detection | Isolation Forest + Z-score |
| Decision | `src/ci_gate/gate.py` | Threshold scoring → pass/fail | rule on top-N mean risk |
| Reporting | `src/ci_gate/report.py` | Human-readable output | HTML/JSON generation |
| Orchestration | `scripts/run_pipeline.py` | Wires all 6 stages end-to-end | — |
| Delivery | `.github/workflows/ci.yml` | Test matrix + risk gate + PR feedback | GitHub Actions |

---

## 5. Design rationale

- **Layered, not monolithic**: ingestion / features / model / decision / reporting are separate modules — each independently testable (`tests/test_ingestion.py`, `test_features.py`, `test_models.py`, `test_perf_anomaly.py`) and swappable (e.g. Locust parser could be replaced by JMeter without touching the model layer).
- **Time-split CV, not k-fold**: test results are time-ordered; k-fold would leak future data into training and produce offline metrics the model can't actually hit once it's scoring runs it hasn't seen yet.
- **Random Forest over XGBoost**: `feature_importances_` gives an interpretable answer to "why was this test flagged," which matters when the gate blocks a PR and someone needs a reason, not just a score.
- **Unsupervised where labels don't exist**: flaky-test detection and perf anomalies have no ground-truth labels, so Isolation Forest is used instead of forcing a supervised approach onto a problem that doesn't have one.
- **The gate is a pure function of config**: `risk_threshold` and `top_n_tests` live in `config.yaml`, not hardcoded — same codebase tunable per team/repo without a code change.
- **CI is the actual product surface**: the ML output isn't a dashboard nobody looks at — it's wired directly into the PR merge decision and posts a comment, so the model's output has to earn its keep at the point where engineers actually work.

## 6. Known limitations (not yet solved)

- **Validated on synthetic data only.** The failure predictor and anomaly detector have never been run against a real project's accumulated CI/perf history — only against `scripts/generate_sample_data.py` output, which has cleaner, hand-injected patterns than real flakiness.
- **No drift monitoring.** Nothing tracks whether a deployed model's accuracy degrades as the test suite and codebase evolve; there's no retraining trigger.
- **Gate averages top-N risk.** `gate.py` compares `mean(top_10_risk_scores)` to the threshold, so one severely at-risk test can be diluted by nine lower-risk ones and still pass. Worth pairing with a hard per-test ceiling.
- **Rolling z-score features self-normalize.** A performance regression that persists longer than `baseline_window` samples gradually gets absorbed into its own rolling baseline and the anomaly score decays back toward zero — a known weakness of any purely rolling reference window.
