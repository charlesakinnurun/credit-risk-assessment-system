# Credit Risk Scoring — Probability of Default

[![CI](https://img.shields.io/badge/CI-ruff%20%7C%20mypy%20%7C%20pytest-2ea44f)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A production-oriented machine learning system that estimates the probability
that a credit-card client will **default** on their next payment. The system is
built end-to-end: leakage-safe data handling, financially meaningful feature
engineering, an interpretable-to-boosted model progression, probability
calibration, business-cost threshold optimisation, explainability,
responsible-ML auditing, a persisted inference artifact, a FastAPI service, a
Streamlit dashboard, and CI.

> **Decision support, not an oracle.** The model estimates a statistical
> association from historical data. It does **not** establish causation, and
> every adverse credit decision requires qualified human review and compliance
> with applicable fair-lending law.

---

## Table of contents

- [Key results](#key-results)
- [Problem statement](#problem-statement)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Data dictionary](#data-dictionary)
- [Data quality](#data-quality)
- [Leakage prevention](#leakage-prevention)
- [Feature engineering](#feature-engineering)
- [Model development](#model-development)
- [Experiment results](#experiment-results)
- [Evaluation metrics (held-out test set)](#evaluation-metrics-held-out-test-set)
- [Class imbalance](#class-imbalance)
- [Threshold selection](#threshold-selection)
- [Probability calibration](#probability-calibration)
- [Explainability](#explainability)
- [Responsible ML](#responsible-ml)
- [API](#api)
- [Dashboard](#dashboard)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Experiment tracking](#experiment-tracking)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)
- [License](#license)

---

## Key results

All numbers below were produced by the committed pipeline (`random_state=42`)
and are reproduced verbatim from `reports/` and `models/model_metadata.json`.

| Metric (held-out test set) | Value |
| --- | --- |
| Model | LightGBM (gradient-boosted trees) |
| ROC-AUC | **0.7774** |
| PR-AUC (average precision) | **0.5315** |
| Calibrated Brier score | **0.1357** (vs 0.1769 uncalibrated) |
| Operating threshold | 0.17 (validation, cost-optimal) |
| Precision / Recall @ operating | 0.365 / 0.758 |
| F1 @ operating | 0.493 |
| F1-optimal point (threshold 0.30) | Precision 0.512 / Recall 0.575 / F1 0.542 |

The model substantially beats a stratified dummy baseline (PR-AUC 0.230) and the
gradient-boosted ensemble earns its extra complexity over logistic regression
(validation PR-AUC 0.566 vs 0.534).

---

## Problem statement

Given a client's six-month credit-card repayment history, predict the
probability that they will default next month (**binary classification**,
`1 = default`, `0 = no default`) and translate it into a lending outcome:

```json
{ "default_probability": 0.72, "risk_tier": "HIGH", "decision": "DECLINE" }
```

### Why credit risk?

Probability-of-default (PD) modelling is a canonical, high-stakes applied ML
problem. It forces the hard engineering and statistical decisions that separate
a portfolio project from a notebook:

- **Class imbalance** — defaults are a minority (22% here); accuracy is
  misleading, so PR-AUC, recall, and calibration matter.
- **Asymmetric costs** — approving a defaulter (false negative) and declining a
  good client (false positive) have very different costs, so the threshold must
  be chosen deliberately, not defaulted to 0.5.
- **Calibration** — a *probability* is only useful if "0.30" really means ~30%.
- **Explainability & fairness** — credit decisions are regulated; scores must be
  explainable, sensitive attributes excluded, and outcomes audited across groups.
- **Leakage** — the easiest way to get an impressive-but-useless result.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
  data/raw/cc_default.xls│  Ingestion → Validation → Cleaning            │
                         └───────────────────────┬──────────────────────┘
                                                 │
                         ┌───────────────────────▼──────────────────────┐
                         │  Stratified 70/15/15 split (target only)      │
                         └───────┬───────────────┬──────────────┬────────┘
                                 │ train         │ validation   │ test (untouched)
                                 ▼               ▼              ▼
                 ┌───────────────────────┐  ┌───────────────┐  ┌───────────────┐
                 │ Preprocessing pipeline│  │ Calibration   │  │ Final         │
                 │ engineering → impute  │  │ + threshold   │  │ evaluation    │
                 │ → winsorise → scale   │  │ selection     │  │ (once)        │
                 │ → one-hot encode      │  └───────┬───────┘  └───────────────┘
                 └───────────┬───────────┘          │
                             ▼                      │
        Baseline + tuned (LR / DT / RF / LightGBM)  │
                 model selection (PR-AUC)           │
                             ▼                      ▼
                 ┌─────────────────────────────────────────────────┐
                 │  CreditRiskModel (single persisted artifact)     │
                 │  preprocessing + classifier + calibrator +       │
                 │  thresholds + reference medians + metadata       │
                 └───────────────┬─────────────────┬───────────────┘
                                 ▼                 ▼
                          FastAPI /predict    Streamlit dashboard
                          /explain /health    local explanations
```

---

## Dataset

**UCI Machine Learning Repository — *Default of Credit Card Clients***
(Yeh & Lien, 2009). Source:
<https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>.

| Property | Value |
| --- | --- |
| Rows | 30,000 |
| Columns | 25 (1 identifier + 23 features + 1 target) |
| Defaults | 6,636 (**22.12%**) |
| Missing values | 0 |
| Duplicate rows / IDs | 0 / 0 |
| Content fingerprint | `dbc4116793a9da0b` |
| Time span | six monthly statements ending one month before the label month |

The raw file is **not committed** (financial data should not live in version
control). It is downloaded on demand by `python -m credit_risk.data.ingestion`
or placed manually at `data/raw/cc_default.xls`.

**Is this the right dataset?** For a *credit-card* default task, yes. It is a
real, widely used benchmark with genuine class imbalance and a clean
codebook. It is **not** a general loan-application dataset: it has no income,
employment, or bureau-score columns. We therefore do **not** fabricate features
such as `debt_to_income_ratio` or `credit_score` — they are not supported by the
data. The API schema reflects the real columns only.

---

## Data dictionary

| Raw column | Project name | Type | Description | Modelling use |
| --- | --- | --- | --- | --- |
| ID | `id` | int | Client identifier | **Excluded** (identifier) |
| LIMIT_BAL | `limit_bal` | float | Credit limit (NT$) | Feature |
| SEX | `sex` | cat | 1 = male, 2 = female | **Audit only** (protected) |
| EDUCATION | `education` | cat | 1 graduate, 2 university, 3 high school, 4+ other | Feature |
| MARRIAGE | `marriage` | cat | 1 married, 2 single, 3 other | **Audit only** (protected) |
| AGE | `age` | int | Age in years | Feature |
| PAY_0 | `pay_0` | ord | Repayment status, most recent month | Feature |
| PAY_2 … PAY_6 | `pay_2` … `pay_6` | ord | Repayment status, 2–6 months ago | Feature |
| BILL_AMT1 … 6 | `bill_amt1` … `bill_amt6` | float | Statement balance (NT$) | Feature |
| PAY_AMT1 … 6 | `pay_amt1` … `pay_amt6` | float | Amount repaid last month (NT$) | Feature |
| default payment next month | `default_payment_next_month` | binary | **Target**: 1 = default | Target |

Repayment-status codes: `-2` no consumption, `-1` paid in full, `0` revolving
credit, `1…9` months of delay. (The raw file names the most recent month `PAY_0`
and omits a `PAY_1`; this is a quirk of the original spreadsheet.)

---

## Data quality

Findings from `validate_data` (written to `reports/data_quality_report.json`):

- **No missing values** and **no duplicate rows or identifiers**.
- **Out-of-codebook categorical codes** exist and are mapped to explicit
  “other” bins rather than dropped: `EDUCATION ∈ {0, 5, 6}` → 4 (`other`),
  `MARRIAGE = 0` → 3 (`other`).
- **No temporal ordering** — the data is a cross-sectional snapshot — so a
  stratified random split is the correct methodology (documented below).
- Extreme bill/payment outliers are **winsorised** at training-set percentiles.

### Data-quality risks

| Risk | Handling |
| --- | --- |
| Out-of-codebook categoricals | Mapped to `other`; validated & unit-tested |
| Skewed amounts / outliers | Winsorised at 1st/99th train percentiles |
| Mild class imbalance | PR-AUC primary metric; class weights compared empirically |
| Single-snapshot data | Cannot measure temporal drift; see Limitations |

---

## Leakage prevention

Leakage is treated as a first-class concern:

1. **Fit-on-train only.** Every learned statistic (imputation medians,
   winsorisation quantiles, scaler parameters, one-hot vocabulary) is fitted
   **exclusively** on the training split and reused for validation, test, and
   live inference. This is enforced structurally — they live inside one
   scikit-learn `Pipeline`.
2. **Stateless feature engineering.** Engineered features are computed row-wise
   (no cross-row aggregates), so no future information can bleed in.
3. **Train/validation/test discipline.** The validation split is used for
   calibration and threshold selection; the **test split is touched exactly
   once**, after all choices are frozen. No hyper-parameter or threshold choice
   ever sees the test set.
4. **No target leakage.** The target is never a feature. Identifier (`id`) and
   protected attributes (`sex`, `marriage`) are excluded from model inputs.
5. **No duplicate-client leakage.** Every `id` is unique; there are no repeated
   clients across splits.
6. **Split validation.** Stratification guarantees the minority class keeps its
   prevalence in every fold.

A regression test (`tests/test_preprocessing.py`) proves the winsoriser uses
training bounds only and that imputation uses the training median, not the
validation distribution.

---

## Feature engineering

Seventeen row-wise, financially meaningful features are added on top of the 21
raw feature columns (production code lives in
`src/credit_risk/features/engineering.py`, never in the notebook):

| Feature | Rationale |
| --- | --- |
| `payment_delay_rate` | Share of months with any delay — repeated delinquency signal |
| `serious_delay_rate` | Share of months with delay ≥ 2 months |
| `max_delay_months` | Worst delay observed in the window |
| `pay_duly_rate` | Share of months paid in full |
| `revolving_use_rate` | Share of months carrying a balance → cash-flow stress |
| `total_bill_amount` / `avg_bill_amount` | Absolute exposure |
| `bill_volatility` | Instability of statement balances |
| `total_payment_amount` / `avg_payment_amount` | Absolute repayment capacity |
| `payment_volatility` | Instability of repayments |
| `net_cash_flow` | Total repaid − total billed |
| `payment_to_bill_ratio` | Core repayment-capability ratio |
| `credit_utilization` | Avg statement balance ÷ credit limit (bureau analogue) |
| `payment_to_limit_ratio` | Repayment relative to available credit |
| `bill_trend` | Whether spending grew over the window |
| `recent_delay_flag` | Most recent month's delinquency status |

**Handling of data pathologies:** missing values → median/most-frequent
imputation; skew/outliers → winsorisation; scaling → standardisation (numeric);
rare/unseen categories → mapping to `other` + `handle_unknown="ignore"`

---

## Model development

A deliberate progression demonstrates whether complexity actually pays off.
Everything is tuned on the training split (inner cross-validation) and scored on
validation; the test set stays untouched.

1. **DummyClassifier** (stratified) — the floor.
2. **Logistic Regression** — interpretable linear baseline.
3. **Decision Tree** — single non-linear tree.
4. **Random Forest** — bagged trees.
5. **LightGBM** — gradient-boosted trees.

Hyper-parameters are searched with `RandomizedSearchCV` scored on
**average precision (PR-AUC)** — the right objective under class imbalance.
The final model is chosen on validation PR-AUC, with a documented
**simplicity tie-break**: if a simpler model is within 0.005 PR-AUC of the best,
the simpler one wins (the regulated-credit preference for an interpretable
model when performance is comparable).

---

## Experiment results

Validation-split comparison (`reports/model_comparison_validation.csv`,
threshold 0.5):

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| DummyClassifier | 0.5226 | 0.2301 | 0.2554 | 0.257 | 0.254 |
| LogisticRegression (default) | 0.7690 | 0.5339 | 0.4593 | 0.662 | 0.352 |
| DecisionTree (default) | 0.6080 | 0.2856 | 0.3905 | 0.385 | 0.396 |
| RandomForest (default) | 0.7541 | 0.5288 | 0.4610 | 0.626 | 0.365 |
| LightGBM (default) | 0.7690 | 0.5377 | 0.4455 | 0.643 | 0.341 |
| tuned LogisticRegression | 0.7689 | 0.5337 | 0.4592 | 0.665 | 0.351 |
| tuned RandomForest | 0.7782 | 0.5519 | 0.5361 | 0.488 | 0.594 |
| **tuned LightGBM (selected)** | **0.7831** | **0.5658** | 0.5331 | 0.466 | 0.623 |

Selected hyper-parameters:

```text
LogisticRegression : C=8.34, class_weight=None
RandomForest       : max_depth=8, min_samples_leaf=2, max_features=sqrt,
                     class_weight=balanced_subsample
LightGBM (final)   : num_leaves=31, learning_rate=0.0168, n_estimators=300,
                     min_child_samples=10, subsample=0.9, colsample_bytree=1.0,
                     reg_lambda=1.0, class_weight=balanced
```

---

## Evaluation metrics (held-out test set)

Metrics are reported on the untouched test split at the frozen operating
threshold **and**, for transparency, at the F1-optimal threshold (regenerate with
`python -m credit_risk.models.evaluate --split test`).

| Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | FPR | FNR | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0.17 (operating)** | 0.365 | 0.758 | 0.493 | 0.7774 | 0.5315 | 0.625 | 0.375 | 0.242 | 0.1357 |
| 0.30 (F1-optimal) | 0.512 | 0.575 | 0.542 | 0.7774 | 0.5315 | 0.844 | 0.156 | 0.425 | 0.1357 |

Confusion matrix at the operating threshold: `TP=755, TN=2191, FP=1313, FN=241`.

**False positives vs. false negatives.** A false negative (approving a
defaulter) is treated as **5× more costly** than a false positive (declining a
good applicant). That cost model drives the threshold down and trades precision
for recall — the system flags more applicants for review/decline to catch more
defaults. The F1-optimal row shows the more balanced alternative; the operating
point is a **business policy choice**, exposed in `configs/config.yaml`.

All ROC / PR / calibration / confusion / threshold figures are written to
`reports/figures/`.

---

## Class imbalance

The target is mildly imbalanced (22% default). We **do not** apply SMOTE by
default: synthesising minority points in a space of discrete repayment-status
codes is hard to justify, and it must never run before the split. Instead:

- The search space includes `class_weight ∈ {None, balanced}` for every family.
- For the selected family this was measured explicitly on validation PR-AUC:

  ```
  unweighted             : 0.5566
  class_weight=balanced  : 0.5658   ← chosen
  ```

- Imbalance is also handled at the **decision boundary** via threshold
  optimisation, and the raw score distortion from class weighting is corrected
  by calibration.

---

## Threshold selection

The default 0.5 threshold is not used. Two operating points are computed on
**validation** data and both reported (`reports/threshold_selection.json`):

- **F1-optimal:** 0.25 (maximises F1 on validation).
- **Business-cost-optimal:** 0.17 — minimises
  `5·FN + 1·FP` — **selected** as the operating point.

The cost model is a documented policy assumption, not a measured quantity. The
threshold is learned once on validation and evaluated once on test.

---

## Probability calibration

Because the output is a *probability*, calibration is evaluated directly
(`reports/calibration_report.json`):

| Method | Validation Brier | Test Brier |
| --- | --- | --- |
| Uncalibrated | — | 0.1769 |
| Sigmoid (Platt) | 0.1354 | — |
| **Isotonic** | **0.1329** | **0.1357** |

The method is selected by **validation** Brier score (isotonic), and calibration
cuts the test Brier score by ~0.041 (−23%). A calibration curve is saved to
`reports/figures/calibration_curve.png`.

---

## Explainability

Two views are provided, both described as *model association*, never causation.

**Global** — model-agnostic permutation importance (ROC-AUC drop when a raw
feature is shuffled) plus the model's native importances
(`reports/permutation_importance.csv`). Top features on the test set:

| Rank | Feature | ROC-AUC drop |
| --- | --- | --- |
| 1 | repayment status, most recent month (`pay_0`) | 0.075 |
| 2 | statement balance, month 1 (`bill_amt1`) | 0.019 |
| 3 | credit limit (`limit_bal`) | 0.016 |
| 4 | amount repaid, month 3 (`pay_amt3`) | 0.011 |
| 5 | repayment status, 2 months ago (`pay_2`) | 0.010 |

**Local** — for a single applicant, each feature is reset to the portfolio
median and the change in the calibrated probability is reported (ablation).
A **positive** contribution means the applicant's actual value raises the score
relative to the median. This is model-agnostic and needs no extra dependency.
Real output for the high-risk applicant above (`pay_0=3`, high utilisation, low
repayments):

```text
Prediction: HIGH RISK     Default probability: 0.72
Risk-increasing : repayment status (month -0), amount repaid (month -3), amount repaid (month -1)
Protective      : age, statement balance (month -6)
```

SHAP is supported as an optional extra (`pip install shap`); when installed it
replaces the ablation view and a beeswarm summary is saved. **Caveat:** these
methods explain what the model learned, not what causes default. No causal claim
is made.

---

## Responsible ML

`sex` and `marriage` are **excluded from the model** (ECOA-protected marital
status; sex is a protected class). `age` and `education` are used as features
but their outcomes are audited. Sensitive attributes are retained **only** to
measure whether errors fall unevenly across groups
(`reports/fairness_*.csv`). At the operating threshold:

| Attribute | Group | n | Recall | FPR | FNR |
| --- | --- | --- | --- | --- | --- |
| sex | female | 2734 | 0.776 | 0.357 | 0.224 |
| sex | male | 1766 | 0.737 | 0.404 | 0.263 |
| age band | ≤25 | 560 | 0.846 | 0.499 | 0.154 |
| age band | 36–45 | 1278 | 0.739 | 0.359 | 0.261 |

Max–min spreads (`reports/fairness_disparity_summary.csv`): recall 3.9 pp and
FPR 4.8 pp across sex; recall 10.7 pp and FPR 15.2 pp across age bands.

**Interpretation and limitations.** These are *descriptive* disparities, not a
verdict. Larger age-band gaps and the very small `marriage="other"` group
(n=54) make some estimates high-variance. Equal group metrics would **not** by
themselves prove fairness, and bias from attributes absent here (race,
ethnicity, disability, income) cannot be measured. The audit is a screening
tool; production use requires human review of every adverse decision.

---

## API

FastAPI service (`app/api.py`).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness + whether the model artifact is loaded |
| GET | `/model-info` | Version, dataset fingerprint, threshold, calibration, test metrics |
| POST | `/predict` | Calibrated probability, risk tier, decision, risk score |
| POST | `/explain` | Ablation-based local risk/protective factors |

**Request schema note.** The service accepts the dataset's **actual** features.
Generic fields such as `annual_income` or `credit_score` are **rejected**
(`extra="forbid"`) because they do not exist in this dataset — inventing them
would produce meaningless predictions.

Real low-risk example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "limit_bal": 120000, "age": 33, "education": 2,
  "pay_0": -1, "pay_2": -1, "pay_3": 0, "pay_4": 0, "pay_5": -2, "pay_6": -2,
  "bill_amt1": 5250, "bill_amt2": 4305, "bill_amt3": 4953,
  "bill_amt4": 4690, "bill_amt5": 4000, "bill_amt6": 3972,
  "pay_amt1": 5250, "pay_amt2": 4305, "pay_amt3": 4953,
  "pay_amt4": 4690, "pay_amt5": 4000, "pay_amt6": 3972
}'
```

```json
{ "default_probability": 0.0717, "risk_tier": "LOW", "decision": "APPROVE", "risk_score": 928, "model_version": "1.0.0+..." }
```

A high-risk applicant (`pay_0=3`, high utilisation, low payments) returns:

```json
{ "default_probability": 0.7197, "risk_tier": "HIGH", "decision": "DECLINE", "risk_score": 280, "model_version": "1.0.0+..." }
```

Validation errors return a structured `422 {"error": "invalid_request", ...}`.
Internal details are never leaked.

---

## Dashboard

A Streamlit dashboard (`app/dashboard.py`) lets a reviewer enter an applicant's
recent statement history and see the calibrated probability, risk tier,
decision, an ablation-based local explanation, global feature importance, and
the model metadata — with an explicit decision-support disclaimer. It is
deliberately sober, not flashy: the ML is the product.

---

## Project structure

```
codealpha-credit-scoring-model/
├── configs/config.yaml            # single source of configuration
├── src/credit_risk/
│   ├── config.py                  # typed config loader (YAML + env overrides)
│   ├── data/                      # ingestion, validation, preprocessing
│   ├── features/engineering.py    # row-wise financial features
│   ├── models/                    # train, evaluate, calibration, predict
│   ├── explainability/            # global + local explanations
│   ├── responsible/fairness.py    # group-metric auditing
│   ├── tracking/tracker.py        # local + optional MLflow experiment tracking
│   └── utils/logging.py           # structured logging
├── app/
│   ├── api.py                     # FastAPI service
│   └── dashboard.py               # Streamlit dashboard
├── tests/                         # 43 pytest tests
├── notebooks/01_eda.ipynb         # exploration only (imports the package)
├── reports/                       # metrics, figures, fairness (generated)
├── models/                        # persisted artifact (generated, git-ignored)
├── Dockerfile, docker-compose.yml
├── .github/workflows/ci.yml
├── pyproject.toml, requirements.txt, requirements-dev.txt
└── Makefile
```

---

## Installation

```bash
git clone <repository-url>
cd codealpha-credit-scoring-model

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install --no-deps -e .        # install the credit_risk package (editable)
```

Requires **Python 3.11+** (developed on 3.14).

---

## Usage

```bash
# 1. Acquire the dataset (or place data/raw/cc_default.xls manually)
python -m credit_risk.data.ingestion

# 2. Train end-to-end (fit, tune, calibrate, threshold, evaluate, persist)
python -m credit_risk.models.train

# 3. Score a demo applicant from the CLI
python -m credit_risk.models.predict

# 4. Audit the saved model on the held-out test split
python -m credit_risk.models.evaluate --split test

# 5. Serve the API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# 6. Launch the dashboard
streamlit run app/dashboard.py
```

Equivalent `make` targets: `make train`, `make evaluate`, `make api`,
`make dashboard`, `make test`, `make lint`, `make typecheck`.

---

## Testing

43 tests over the critical paths — data validation, cleaning, feature
engineering, leakage safety, preprocessing, metrics, threshold logic, the
inference artifact, probability bounds, and the API contract (including
invalid-input and unknown-field rejection).

```bash
pytest -q
```

Tests use **synthetic** data, so they run in CI without the real (git-ignored)
dataset. Key invariants asserted include `0 ≤ default_probability ≤ 1`,
`0 ≤ risk_score ≤ 1000`, `decision ∈ {APPROVE, REVIEW, DECLINE}`, and that
fit-time statistics never leak from validation.

---

## Docker

The image runs as a **non-root** user, uses a multi-stage build for a lean
runtime, and ships a dependency-free health check.

```bash
# Train inside the stack (mounts data/, models/, reports/)
docker compose --profile train up train

# Start the API (http://localhost:8000/docs) and dashboard (http://localhost:8501)
docker compose up --build

# Optional MLflow server on :5000
docker compose --profile mlflow up
```

The trained artifact is shared through the `./models` volume, so the API and
dashboard always load the exact model the training service produced.

---

## CI/CD

`.github/workflows/ci.yml` runs on every push/PR:

```
install → ruff (lint) → pytest → mypy (type-check) → docker build
```

CI fails on lint errors, test failures, or type errors.

---

## Experiment tracking

Every training run writes a self-contained JSON record (parameters, metrics,
dataset fingerprint, feature configuration, artifact paths, git commit) to
`reports/experiments/<run_id>.json` and appends to `runs.jsonl`, making the best
run reproducibly identifiable without a tracking server. If `mlflow` is
installed and `CREDIT_RISK_MLFLOW=1` is set, the same payload is mirrored to
MLflow; a `docker compose` MLflow profile is provided.

---

## Reproducibility

- One **global seed** (`seed: 42`, overridable via `CREDIT_RISK_SEED`) drives the
  split, model fitting, and search.
- Configuration is centralised in `configs/config.yaml`.
- The dataset is versioned by a content **fingerprint** recorded on the model.
- The **complete** inference path (engineering, imputation, winsorisation,
  scaling, encoding, classifier, calibrator, thresholds) is persisted as one
  artifact — training and serving cannot drift.
- The training run is deterministic: re-running `python -m credit_risk.models.train`
  reproduces the metrics reported above.

```bash
git clone <repository-url> && cd codealpha-credit-scoring-model
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt && pip install --no-deps -e .
pytest
python -m credit_risk.data.ingestion
python -m credit_risk.models.train
```

---

## Limitations

- **Dataset scope.** Credit-card repayment history only — no income, employment,
  bureau score, or macroeconomic features. Results do not transfer to a general
  loan portfolio without retraining on appropriate data.
- **No temporal validation.** A single cross-sectional snapshot cannot support
  out-of-time evaluation or drift measurement; the stratified random split is
  the honest choice here, but real deployments need time-based backtesting.
- **Modest absolute performance.** PR-AUC ≈ 0.53 reflects a genuinely hard,
  noisy signal, not a tuning failure; the model is useful for *ranking and
  review*, not for high-confidence automated decisions.
- **Fairness analysis is incomplete.** Protected attributes present here are few;
  disparities in unobserved groups cannot be measured.
- **Local explanations** (ablation) are computationally simple and describe the
  model, not causal effects.
- **Calibration** is fitted on a modest validation set (4,500 rows); isotonic
  can overfit small calibration sets in principle.

---

## Future improvements

- Out-of-time validation on a rolling temporal dataset.
- Optuna-based search with pruning and a persisted study.
- Monotonic constraints on engineered risk features for stronger regulatory
  defensibility.
- Cost-sensitive learning / expected-profit optimisation instead of F1/cost
  heuristics.
- Model registry + canary/blue-green rollout and drift monitoring in production.
- Deeper fairness work: intersectional analysis, equalised-odds post-processing,
  and calibrated group thresholds where lawful.
- Feature-store integration to guarantee online/offline parity.

---

## License

Released under the [MIT License](LICENSE).

<!-- https://opncd.ai/share/OQTxkofa -->