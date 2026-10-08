# SecurePredict

**Explainable Industrial IoT Monitoring with Predictive Maintenance and Network Threat Detection**

SecurePredict is a research/benchmark demonstration that combines two independent Industrial IoT monitoring problems under one application:

1. **Machine Health** — predict machine-failure risk from the AI4I 2020 predictive-maintenance benchmark using XGBoost.
2. **Network Security** — classify network-flow traffic as normal or attack from RT-IoT2022 using a numeric-flow Random Forest.
3. **Explainability** — use SHAP to show which features influence model behavior.
4. **Application Layer** — expose the frozen models through a reusable inference layer and a Streamlit dashboard.

> **Important:** SecurePredict is a benchmark/research demonstration. The reported metrics are held-out results on public datasets and are not guarantees of real-world factory or network performance.

---

## Architecture

```text
                         SECUREPREDICT
                               |
                +--------------+--------------+
                |                             |
         Machine Health                 Network Security
                |                             |
          AI4I 2020 data                RT-IoT2022 data
                |                             |
            XGBoost                    Random Forest
                |                             |
          failure risk                 attack risk
                |                             |
                +--------------+--------------+
                               |
                            SHAP XAI
                               |
                       Reusable Inference
                               |
                        Streamlit Dashboard
```

The two datasets are modeled independently. Their rows are **not merged into one training table**.

---

## Completed experimental workflow

### V0.1 — Data integrity and split policy
- Audited missing values and duplicates.
- Identified conflicting duplicate-label groups in RT-IoT2022.
- Removed 128 ambiguous rows from RT-IoT2022.
- Built train/validation/test splits.
- Prevented duplicate groups from crossing RT-IoT2022 split boundaries.

### V0.1.1 — Feature quality
- AI4I: 6 candidate features, no missing values, no constant features.
- RT-IoT2022: identified one constant feature and several near-constant features.
- Removed the constant `bwd_URG_flag_count` from the final RT-IoT modeling path.

### V0.1.2 / V0.3 — Baselines
Compared dummy, Logistic Regression, Random Forest, and XGBoost baselines.

### V0.2 — RT-IoT2022 feature ablation
Tested removal of categorical, port, and related inputs.

The final security model uses **78 numeric network-flow features**, excluding:
- `bwd_URG_flag_count`
- `proto`
- `service`
- `id.orig_p`
- `id.resp_p`

### V0.4 — AI4I XGBoost tuning
Randomized hyperparameter search with 4-fold StratifiedKFold and PR-AUC/average precision as the selection score.

### V0.4.1 — Threshold study
AI4I operating thresholds were selected on validation data:
- **0.77** — primary failure decision.
- **0.07** — higher-recall early-warning decision.

Thresholds were frozen before the untouched test evaluation.

### V0.5 — Final held-out evaluation

#### AI4I 2020 — final XGBoost
Primary threshold = 0.77

| Metric | Held-out test |
|---|---:|
| ROC-AUC | 0.9757 |
| PR-AUC | 0.8301 |
| F1 | 0.7879 |
| Recall | 0.7647 |
| Precision | 0.8125 |
| Accuracy | 0.9860 |

#### RT-IoT2022 — final Random Forest
Numeric-flow-only model

| Metric | Held-out test |
|---|---:|
| ROC-AUC | 0.999974 |
| PR-AUC | 0.999997 |
| F1 | 0.9992 |
| Recall | 0.9989 |
| Precision | 0.9995 |

The RT-IoT2022 results are unusually strong on this benchmark; they should not be interpreted as production-network performance without external validation.

### V0.6 — SHAP explainability
Generated:
- global SHAP feature importance
- representative local SHAP explanations
- AI4I and RT-IoT2022 visualizations and tables

SHAP analysis uses validation data; the held-out test set remains untouched.

### V0.7 — Reusable inference layer
Added schema-aware model loading, validation, AI4I threshold logic, and RT-IoT2022 risk classification.

### V0.8 — Streamlit dashboard
Pages:
- Overview
- Machine Health
- Network Security
- Explainability
- Model Performance
- About

### V0.9 — Automated tests
### V1.0 - Hardening
- Inference rejects NaN, Inf, invalid `Type`, non-numeric and out-of-range values.
- Dashboard explains the user's own input with SHAP (not just static examples).
- Metrics shown in the dashboard are read from the saved report tables.
- AI4I bootstrap confidence intervals (`16_ai4i_uncertainty.txt`) and RT-IoT2022 per-attack-type breakdown (`17_*`).
- GitHub Actions CI, provenance document, stale V0.1 shell removed.

Current test suite (inference, validation, explanations, dashboard smoke test):

```text
33+ passed
```

---

## Project structure

```text
SecurePredict/
├── .github/workflows/ci.yml        # pytest on every push / PR, docker build
├── src/
│   ├── data/                       # one script per experiment stage (V0.1-V1.0.4)
│   ├── inference/
│   │   ├── predictors.py           # predict_* / explain_*
│   │   ├── validation.py           # NaN / Inf / type / range checks
│   │   └── explain.py              # per-input SHAP
│   └── dashboard/app.py            # Streamlit app
├── tests/                          # inference, validation, explain, dashboard
├── models/{final,tuned,baseline,ablation,xgboost_baseline}/
├── reports/v0_1/{tables,figures}/  # numbered reports 02-17
├── EXPERIMENT_PROVENANCE.md        # artifact-level audit trail
├── requirements.txt  Dockerfile  .dockerignore  .gitignore
└── README.md
```

---

## Run locally

### 1. Create and activate an environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run automated tests

```powershell
python -m pytest -q
```

### 4. Run the inference smoke test

```powershell
python -X utf8 src\inference\smoke_test.py
```

### 5. Run the dashboard

```powershell
streamlit run src\dashboard\app.py
```

---

## Run with Docker

Build:

```powershell
docker build -t securepredict .
```

Run:

```powershell
docker run --rm -p 8501:8501 securepredict
```

Open:

```text
http://localhost:8501
```

The Docker image contains the model artifacts and dashboard code copied from the repository.

---

## Data and modeling notes

### AI4I 2020
Target:
- `Machine failure`

Inputs:
- `Type`
- `Air temperature`
- `Process temperature`
- `Rotational speed`
- `Torque`
- `Tool wear`

Failure-mode columns such as `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` were excluded from model inputs because they are derived from failure information and would introduce target-derived leakage.

### RT-IoT2022
Target:
- binary Normal vs Attack

Normal classes retained:
- `MQTT_Publish`
- `Thing_Speak`
- `Wipro_bulb`

All other retained classes are treated as Attack.

Ambiguous duplicate-label groups were removed before splitting.

---

## Explainability

SHAP is used for:
- **Global explanations** — which features matter most across validation examples.
- **Local explanations** — why a representative sample received its prediction.

The dashboard displays the generated SHAP figures from `reports/v0_1/figures/`.

---

## Engineering principles

SecurePredict intentionally keeps the current implementation modular without over-engineering it:

```text
Model artifacts
      ↓
Inference layer
      ↓
Dashboard
```

The inference layer is isolated from the UI so the same prediction logic can later be exposed through an API or streaming service.

---

## Limitations and next steps

This project does **not** claim to be a production-ready industrial safety or network-defense system.

Production deployment would require:
- external validation on independent data
- telemetry integration
- model/data drift monitoring
- security hardening
- audit logging
- alert management
- latency and load testing
- domain-specific operational validation

Possible future architecture:

```text
MQTT / API ingestion
        ↓
FastAPI inference service
        ↓
time-series / event store
        ↓
monitoring + dashboard
        ↓
drift / performance monitoring
```

---

## Resume-ready summary

> Built an explainable Industrial IoT monitoring system combining predictive maintenance and network-flow threat detection using XGBoost and Random Forest; performed leakage-aware data auditing, validation-based threshold optimization, feature ablation, SHAP explainability, reusable inference, automated testing, and Streamlit application.

