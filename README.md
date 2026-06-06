# InsureIQ v2.2 — AI-Powered Insurance Premium Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.0-lightgrey?style=flat-square&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?style=flat-square&logo=scikit-learn" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Deployed-Render-purple?style=flat-square" alt="Render">
  <img src="https://img.shields.io/badge/Model-GradientBoosting-green?style=flat-square" alt="Model">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="MIT">
</p>

> A production-hardened machine learning application that predicts annual health insurance premiums from demographic, lifestyle, and healthcare history inputs — deployed on Render with fail-fast startup validation, security hardening, and a deep matte-blue multi-step UI.

---

## Project Overview

Health insurance pricing is one of the most consequential and opaque decisions consumers face. Insurers weigh dozens of factors — age, BMI, smoking status, medical history, income — and produce a premium figure that most people cannot anticipate or challenge. InsureIQ demystifies that process.

InsureIQ is a full-stack machine learning web application that takes ten patient-level inputs and predicts their likely annual insurance charge in real time. Beyond the number, it provides a risk profile classification (Low / Medium / High), personalised lifestyle advice, and **what-if simulations** — showing users exactly how much they could save by quitting smoking, reaching a healthy BMI, or comparing their premium at a younger age.

The project was built to demonstrate the complete arc of applied machine learning engineering: dataset acquisition, feature design, model training and validation, API development, security hardening, and cloud deployment — all in a single, self-contained repository.

### Real-World Relevance

- Insurance premiums in the US average over $7,700/year for an individual. A tool that helps people understand and anticipate that cost has direct financial value.
- The project tackles a common ML anti-pattern — **data leakage** — head-on, documents the discovery, and ships a corrected model with honest metrics. This makes it a stronger portfolio piece than one that simply reports inflated numbers.
- The full-stack architecture mirrors production ML systems: a trained model artefact, a validation layer, an API, and a user-facing frontend, each with independent correctness guarantees.

---

## Key Features

| Feature | Detail |
|---|---|
| **Premium prediction** | Gradient Boosting regressor trained on 1,338 real insurance records, extended with four statistically-generated features |
| **Risk profile engine** | Classifies each prediction as Low / Medium / High Risk with a personalised score and actionable advice |
| **What-if simulations** | Live re-prediction showing savings from smoking cessation, healthy BMI, and age comparison |
| **Multi-step form UI** | Four-step guided input wizard with animated progress bar and real-time range sliders |
| **Fail-fast startup** | Model and metadata loaded and validated before Flask binds — bad deploys crash immediately, not silently |
| **Feature schema guard** | Startup comparison of `model_metadata.json` against `FEATURE_ORDER` in `app.py`; mismatch aborts launch |
| **Health endpoints** | `/health` and `/system-status` endpoints for uptime monitoring and deployment diagnostics |
| **Security hardening** | CSP, `X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection`, `Referrer-Policy` on every response |
| **Input validation** | Server-side range checks on every field; typed sanitisation before any model call |
| **Render deployment** | Zero-config `render.yaml` with `buildCommand` that trains the model at deploy time |
| **Deep matte-blue UI** | Custom dark design system with accent blues, glowing progress indicator, risk badge colours, and insight cards |
| **Leakage-free model** | Two data-leaking features (`Claim_Amount`, `Hospital_expenditure`) identified and permanently removed in v2.2 |

---

## Architecture Overview

```
User (Browser)
      │
      ▼
┌─────────────────────────────────────┐
│  Frontend  (templates/index.html)   │
│  Multi-step form · Range sliders    │
│  Risk badge · What-if cards         │
└──────────────┬──────────────────────┘
               │ POST /predict  (JSON)
               ▼
┌─────────────────────────────────────┐
│  Flask API  (app.py)                │
│  Input parsing · Field validation   │
│  REQUIRED_FIELDS check              │
│  Range & type sanitisation          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Startup Validation Layer           │
│  load_model()  → smoke-test pkl     │
│  load_metadata() → schema check     │
│  FEATURE_ORDER match guard          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Prediction Engine                  │
│  sklearn Pipeline                   │
│  StandardScaler → GBR               │
│  predict_premium() · get_risk()     │
│  what-if re-predictions             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Response Layer                     │
│  prediction · risk profile          │
│  whatif scenarios · security hdrs   │
└─────────────────────────────────────┘
```

**Frontend** collects ten inputs across four guided steps and sends a single JSON POST. It renders the result — dollar amount, risk badge, three what-if savings cards, and a plain-English advice line — without a page reload.

**Flask API** performs input parsing and field-level validation before any model call. Missing or malformed fields return structured 400 errors; the model is never called with incomplete data.

**Startup Validation Layer** runs at import time, before the server accepts connections. If the model file is missing, corrupt, or was trained on a different feature schema than the app expects, the process exits with a clear error message.

**Prediction Engine** wraps the sklearn `Pipeline` (scaler + GBR). It handles the FEATURE_ORDER → numpy array assembly, prediction, risk scoring, and three parallel what-if re-predictions in a single request cycle.

**Response Layer** assembles the JSON payload and attaches security headers to every outbound response via an `after_request` hook.

---

## Machine Learning Pipeline

### Dataset

The base dataset is the **Medical Cost Personal Dataset** (Brett Lantz, *Machine Learning with R*), a widely-used benchmark of 1,338 real de-identified US insurance records. It contains seven columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges` (the annual premium — the training target).

Dataset URL: `https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv`

The dataset is downloaded once and cached as `insurance_base.csv`. All subsequent training runs use the cache; the `--offline` flag forces cache-only mode for reproducible builds.

### Feature Engineering

The seven-column base dataset is extended to ten features by generating three additional columns with medically plausible statistical correlations, anchored to real risk factors — not to the target variable.

| Generated Feature | Generation Logic |
|---|---|
| `past_consultations` | Poisson-sampled; rate driven by age, BMI, smoker status, and number of children |
| `num_of_steps` | Normal-sampled; mean decreases for smokers, obese individuals, and older patients |
| `NUmber_of_past_hospitalizations` | Poisson-sampled; rate driven by smoker status, BMI above 30, and age |
| `Anual_Salary` | Normally-sampled around an age-based career curve; weakly correlated with age |

Crucially, **none of these features is derived from `charges`** (the target). This was the root cause of the v2.1 leakage issue, corrected in v2.2.

### Preprocessing

All features pass through a `StandardScaler` as the first step of a `sklearn.pipeline.Pipeline`. Scaling is fit only on training data and applied identically to test data and at inference time — no data leaks through the scaler.

### Train / Test Split

- **Split**: 80% train (1,070 rows) / 20% test (268 rows)
- **Stratification**: stratified on the `smoker` column to preserve the ~20% smoker prevalence in both partitions
- **Random state**: `42` for full reproducibility

### Model

**Algorithm**: `GradientBoostingRegressor` inside a `sklearn.pipeline.Pipeline`

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 400 | Sufficient trees for the dataset size without overfitting |
| `max_depth` | 5 | Captures non-linear interactions without excessive depth |
| `learning_rate` | 0.04 | Conservative rate; pairs with 400 trees for stable convergence |
| `subsample` | 0.85 | Stochastic gradient boosting; reduces variance |
| `min_samples_leaf` | 5 | Prevents overly specific leaf splits |
| `max_features` | 0.8 | Feature subsampling per split; additional regularisation |
| `random_state` | 42 | Reproducibility |

### Cross-Validation

5-fold cross-validation is run on the full dataset (not just the training split) after hold-out evaluation. This gives a generalisation estimate that accounts for variance across different data partitions.

### Serialisation & Metadata

The trained pipeline is serialised with `joblib.dump()` to `insurance_model.pkl`. Training also generates `model_metadata.json`, which records:

- Training timestamp and dataset source
- `FEATURE_ORDER` list and `feature_count`
- Hold-out and cross-validation metrics
- Per-feature importances
- Full feature schema (type and range for each input)

At startup, `app.py` loads this metadata and asserts that its `features` list matches `FEATURE_ORDER` exactly. Any discrepancy aborts the server.

### Inference Workflow

1. JSON payload arrives at `POST /predict`
2. `REQUIRED_FIELDS` presence check
3. `sanitise_float()` on every value
4. Range validation (age 0–120, BMI 0–80, etc.)
5. Feature dict assembled in `FEATURE_ORDER` sequence
6. `np.array(...).reshape(1, -1)` fed to `pipeline.predict()`
7. `get_risk_profile()` scores the prediction
8. Three what-if re-predictions computed (non-smoker, healthy BMI, ten years younger)
9. JSON response returned

---

## Feature List

| # | Feature | Type | Range | Importance | Why It Matters |
|---|---|---|---|---|---|
| 1 | `age` | float | 18–64 | 10.25% | Premium risk increases non-linearly with age; older patients accumulate more health events |
| 2 | `sex` | binary | 0/1 | 0.27% | Biological sex influences certain disease incidences and actuarial risk tables |
| 3 | `bmi` | float | 15–55 | 16.84% | BMI above 30 is strongly associated with diabetes, hypertension, and cardiovascular disease |
| 4 | `children` | int | 0–5 | 1.16% | Dependants increase total family medical expenditure and affect premium tiers |
| 5 | `smoker` | binary | 0/1 | **59.03%** | Smoking is the single strongest predictor; smokers pay 2–4× more on average in this dataset |
| 6 | `past_consultations` | int | 0–20 | 2.00% | Frequent outpatient visits signal chronic conditions that inflate ongoing costs |
| 7 | `num_of_steps` | float | 0–20,000 | 6.08% | Daily activity level is a proxy for general fitness; lower steps correlate with higher health risk |
| 8 | `NUmber_of_past_hospitalizations` | int | 0–10 | 1.05% | Prior hospitalisation history is a direct indicator of elevated future medical spend |
| 9 | `Anual_Salary` | float | $20k–$300k | 2.57% | Income correlates with healthcare access behaviour and indirectly with lifestyle factors |
| 10 | `region` | int | 0–3 | 0.75% | Geographic region affects provider pricing, state regulations, and population health baselines |

Feature importances are extracted from the GBR's `feature_importances_` attribute post-training and recorded in `model_metadata.json`. They sum to 1.0.

---

## Model Performance

| Metric | Value |
|---|---|
| **R²** (hold-out test set) | **0.8573** |
| **Cross-Validation R²** (5-fold) | **0.8402 ± 0.0325** |
| **MAE** | **$2,921.58** |
| **RMSE** | **$4,588.44** |
| Train size | 1,070 records |
| Test size | 268 records |

### Why These Metrics Are Trustworthy

An R² of 0.86 on a 1,338-row insurance dataset is a strong result — and it is an *honest* one. The model is not over-fitted: the hold-out R² (0.8573) and the cross-validation mean (0.8402) are within 2 percentage points of each other, and the CV standard deviation of ±0.0325 indicates stable generalisation across data folds.

The MAE of ~$2,922 means the model's predictions are within roughly $2,900 of the true annual premium on average. Given that premiums in this dataset range from $1,122 to $63,770 — a span of over $62,000 — this represents a mean absolute error of approximately 4.7% of the full range. For a 1,338-row dataset with four synthetic features and no external enrichment, this is a realistic and defensible result.

Contrast this with the v2.1 metrics (R² = 0.93, MAE = $1,444), which appeared stronger but were inflated by `Claim_Amount` — a feature derived directly from the target variable. The v2.2 metrics reflect what the model genuinely learned about insurance risk.

---

## Leakage Audit & Model Integrity

### What Happened in v2.1

The original feature set included two variables that appeared legitimate but were not:

- **`Claim_Amount`**: Generated as `charges × uniform(0.05, 0.55) × normal(1.0, 0.12)`. This is a direct linear transformation of `charges` — the training target — with multiplicative noise. Its Pearson correlation with the target was **r ≈ 0.85**, giving it an R² of ~0.72 when used alone. The model assigned it **29.7% feature importance**.
- **`Hospital_expenditure`**: Generated as a function of past hospitalisation counts *and* `Claim_Amount`, inheriting secondary leakage from the same source. It carried 0.6% feature importance.

Together, these two features consumed **30.3% of the model's explanatory power** and allowed the GBR to learn a shortcut — predicting the target from a noisy copy of itself — rather than learning the true relationships between age, BMI, lifestyle, and premium cost.

### How It Was Detected

A formal leakage audit traced the synthetic data generation code in `train_model.py` against the target variable `charges`. The line:

```python
Claim_Amount = np.clip(charges * claim_frac * claim_noise, 0, 50000)
```

makes the derivation explicit. Once the audit confirmed the correlation coefficient and feature importance share, the fix was unambiguous.

### How It Was Fixed

In v2.2, both features were **permanently removed** from every layer of the stack:

- `train_model.py` — generation code deleted; removed from `FEATURE_ORDER` and `feature_schema`
- `app.py` — removed from `FEATURE_ORDER`, `REQUIRED_FIELDS`, and the `predict()` inputs dict
- `templates/index.html` — form fields deleted; JS `collectData()` payload updated
- `insurance_model.pkl` — retrained from scratch; `n_features_in_ = 10` confirmed
- `model_metadata.json` — regenerated; `feature_count = 10` confirmed

A full post-fix grep across all text files confirmed **zero occurrences** of either feature name in any source file.

### The Impact on Metrics — and Why It Matters

| | v2.1 (leaked) | v2.2 (clean) |
|---|---|---|
| R² | 0.9295 | **0.8573** |
| MAE | $1,444 | **$2,922** |

R² dropped by 7 points. This is not a regression — it is the model being forced to learn from legitimate signals only. The v2.2 model makes predictions that are grounded in age, BMI, smoking status, and medical history rather than in a noisy echo of the answer.

Shipping honest metrics rather than inflated ones is an engineering choice. A model that reports R² = 0.93 due to leakage is worse than one that reports R² = 0.86 without it, because the former will silently degrade the moment the leaking feature is unavailable, sparse, or misreported by users. The v2.2 model is robust to that failure mode.

---

## Security Features

Security headers are applied to every HTTP response via a Flask `after_request` hook. No route is exempt.

| Header | Value | Protection |
|---|---|---|
| `Content-Security-Policy` | `default-src 'self'; script-src 'self' 'unsafe-inline' ...` | Prevents XSS via inline script injection from untrusted origins |
| `X-Frame-Options` | `DENY` | Blocks clickjacking by preventing the page from being embedded in an iframe |
| `X-Content-Type-Options` | `nosniff` | Stops browsers from MIME-sniffing a response away from the declared content type |
| `X-XSS-Protection` | `1; mode=block` | Activates the browser's built-in XSS filter and blocks the page on detection |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Limits referrer information sent to third-party origins |

**Input validation** is enforced at two layers:

1. **Presence check** — `REQUIRED_FIELDS` verified before any parsing
2. **Type sanitisation** — `sanitise_float()` raises a typed `ValueError` on non-numeric input
3. **Range validation** — age, BMI, sex, smoker, children, and region all checked against valid ranges before the model call

**Startup validation** prevents a compromised or mis-versioned model from serving predictions:

- `load_model()` — verifies the pkl file exists, deserialises cleanly, and passes a zero-vector smoke test
- `load_metadata()` — verifies the JSON is parseable and that `features` matches `FEATURE_ORDER` exactly
- Both call `sys.exit(1)` on failure, preventing a degraded or misconfigured server from accepting traffic

---

## Health Monitoring

### `GET /health`

Lightweight liveness probe. Returns 200 if the server is running and the model is loaded.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.2.0"
}
```

### `GET /system-status`

Full diagnostics endpoint. Returns model metadata, runtime metrics, memory usage, and top feature importances.

```json
{
  "status": "healthy",
  "version": "2.2.0",
  "uptime_seconds": 142,
  "model_loaded": true,
  "model_size_kb": 1291,
  "deployment_mode": "production",
  "memory_mb": 214.3,
  "model_metadata": {
    "trained_at": "2026-06-06T07:11:48",
    "dataset_source": "https://raw.githubusercontent.com/...",
    "dataset_size": 1338,
    "algorithm": "GradientBoostingRegressor (sklearn Pipeline + StandardScaler)",
    "feature_count": 10,
    "metrics": {
      "mae": 2921.58,
      "rmse": 4588.44,
      "r2": 0.8573,
      "cv_r2_mean": 0.8402,
      "cv_r2_std": 0.0325
    },
    "top_features": [
      ["smoker", 0.5903],
      ["bmi", 0.1684],
      ["age", 0.1025],
      ["num_of_steps", 0.0608],
      ["Anual_Salary", 0.0257]
    ]
  }
}
```

---

## Project Structure

```
insureiq/
├── app.py                  # Flask application — routing, validation, prediction, security
├── train_model.py          # ML pipeline — data acquisition, feature engineering, training, export
├── insurance_model.pkl     # Trained sklearn Pipeline (generated by train_model.py)
├── model_metadata.json     # Training artefact — metrics, schema, feature importances
├── insurance_base.csv      # Cached base dataset (downloaded from GitHub on first run)
├── requirements.txt        # Pinned Python dependencies
├── render.yaml             # Render deployment configuration
├── README.md               # This file
└── templates/
    └── index.html          # Single-page frontend — multi-step form, results dashboard
```

---

## Installation Guide

### Prerequisites

- Python 3.10 or higher
- pip
- Internet access for initial dataset download (or place `insurance_base.csv` manually)

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/insureiq.git
cd insureiq

# 2. (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model
#    Downloads the dataset, trains GBR, saves insurance_model.pkl and model_metadata.json
python train_model.py

#    Offline mode (if dataset already cached):
python train_model.py --offline

# 5. Start the development server
python app.py
```

The server starts on `http://localhost:5000`. Open that URL in a browser to use the UI.

---

## Render Deployment Guide

InsureIQ ships with a `render.yaml` that fully automates deployment on Render's free tier.

### `render.yaml` Explained

```yaml
services:
  - type: web
    name: insureiq
    runtime: python
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: gunicorn app:app --workers 1 --threads 2 --timeout 120 --bind 0.0.0.0:$PORT --log-level info
    plan: free
    envVars:
      - key: FLASK_ENV
        value: production
      - key: PYTHONUNBUFFERED
        value: "1"
```

**Build phase** installs all dependencies and then runs `train_model.py`, which downloads the dataset, trains the model, and writes `insurance_model.pkl` and `model_metadata.json` to the deployment filesystem.

**Start phase** launches Gunicorn with a single worker and two threads — the optimal configuration for Render's free tier 512 MB RAM ceiling. Two threads allow concurrent requests without the memory overhead of a second worker process.

**Startup validation** runs before Gunicorn accepts connections. If the model or metadata fails validation, the process exits and Render marks the deployment as failed — preventing a broken build from reaching users.

### Deployment Steps

1. Push the repository to GitHub (include `insurance_base.csv` to avoid a network dependency at build time, or rely on the download)
2. Create a new **Web Service** on [render.com](https://render.com) and connect the repository
3. Render auto-detects `render.yaml` — no manual environment configuration required
4. Monitor the build log; the training run logs MAE, RMSE, R², and CV scores
5. Once healthy, the `/health` endpoint confirms readiness

### Resource Profile (Free Tier)

| Resource | Value |
|---|---|
| RAM | ~210–250 MB (model ~1.3 MB, Python runtime ~150 MB) |
| Workers | 1 |
| Threads | 2 |
| Cold start | ~5–8 s |
| Model training at build | ~1 s |

---

## API Documentation

### `POST /predict`

Accepts a JSON body with ten fields. Returns a prediction, risk profile, and what-if scenarios.

#### Request

```http
POST /predict
Content-Type: application/json
```

```json
{
  "age": 35,
  "sex": 1,
  "bmi": 28.5,
  "children": 2,
  "smoker": 0,
  "past_consultations": 4,
  "num_of_steps": 7500,
  "past_hospitalizations": 1,
  "annual_salary": 75000,
  "region": 2
}
```

| Field | Type | Range | Notes |
|---|---|---|---|
| `age` | float | 0–120 | Patient age in years |
| `sex` | int | 0 or 1 | 0 = female, 1 = male |
| `bmi` | float | 0–80 | Body mass index |
| `children` | int | 0–10 | Number of dependants |
| `smoker` | int | 0 or 1 | 0 = non-smoker, 1 = smoker |
| `past_consultations` | int | 0–20 | Outpatient visits in past year |
| `num_of_steps` | float | 0–20,000 | Average daily step count |
| `past_hospitalizations` | int | 0–10 | Number of prior hospital admissions |
| `annual_salary` | float | 20,000–300,000 | Annual income in USD |
| `region` | int | 0–3 | 0=northeast, 1=northwest, 2=southeast, 3=southwest |

#### Response — Success (200)

```json
{
  "prediction": 12847.32,
  "risk": {
    "level": "Medium Risk",
    "color": "medium",
    "icon": "🟡",
    "score": 30,
    "advice": "Moderate risk profile. Regular checkups and a healthy BMI can help reduce costs."
  },
  "whatif": {
    "nonsmoker": null,
    "healthy_bmi": 10234.18,
    "age_diff": 9821.44,
    "current_age": 35,
    "current_bmi": 28.5,
    "is_smoker": false
  }
}
```

#### Response — Validation Error (400)

```json
{
  "error": "Missing required fields: smoker, region"
}
```

```json
{
  "error": "Validation error: BMI must be 0–80"
}
```

### `GET /health`

Returns `200 OK` when the server is running and the model is loaded.

### `GET /system-status`

Returns full runtime diagnostics including uptime, memory, model metrics, and top features. See [Health Monitoring](#health-monitoring) for a full response example.

---

## UI Showcase

InsureIQ's frontend is a single-page application built with vanilla HTML, CSS, and JavaScript — no frontend framework required.

### Design System

The interface uses a **deep matte-blue dark theme** built on CSS custom properties. The primary background sits at near-black `#07111f`, layered with subtle radial glows and a fine dot-grid texture. Accent colours are drawn from a blue ramp (`#2d7ff9` primary, `#4da3ff` highlight), with semantic overrides for success (green), warning (amber), and danger (red) states — used exclusively for the risk badge.

### Multi-Step Form

The input form is divided into four clearly labelled steps, navigated by Previous / Next buttons and a progress bar that animates from 25% to 100% as the user advances:

- **Step 1 — Personal Profile**: Age, sex, number of children, smoking status
- **Step 2 — Physical Health**: BMI (with live slider), past consultations, daily steps
- **Step 3 — Medical History**: Past hospitalisations
- **Step 4 — Financial Profile**: Annual salary, region

All numeric inputs use range sliders with real-time value display. Toggle buttons replace dropdowns for binary fields (sex, smoker, region). The form preserves values on back-navigation.

### Results Dashboard

After prediction the page scrolls to a results panel containing:

- **Premium figure** — large, formatted dollar amount
- **Risk badge** — colour-coded Low / Medium / High with an emoji indicator
- **What-if insight cards** — up to three cards showing projected savings from smoking cessation, reaching BMI < 25, or comparing to a ten-years-younger baseline
- **Personalised advice** — a plain-English recommendation based on the risk profile
- **Recalculate button** — returns to the form without a page reload

### Screenshots

> *Add screenshots here after deployment.*
>
> Suggested captures:
> - Step 1 form (desktop)
> - Step 3 form with sliders active
> - Results panel with High Risk badge and what-if cards
> - Mobile view of the multi-step form

---

## Challenges Faced

### Synthetic Feature Design Without Leakage

Extending a 7-column real dataset to 10 columns required generating synthetic features that are statistically plausible but causally independent of the target. The first attempt introduced `Claim_Amount` and `Hospital_expenditure`, both of which were inadvertently derived from `charges`. Identifying this required tracing every feature's generation code back to its source variables — a process that revealed the leakage only when the dependency chain was written out explicitly.

The lesson: synthetic feature generation must be validated not just for statistical realism but for causal independence from the target. Correlation checks between generated features and `y` should be a mandatory step in any pipeline that augments real data.

### Honest Metrics vs. Impressive Metrics

The corrected model's R² of 0.86 is lower than the leaked model's 0.93. There was a temptation to present the higher number, particularly for a portfolio context. The decision to document the leakage, fix it, and ship the lower-but-honest metrics was deliberate — it demonstrates exactly the kind of engineering judgement that separates a careful practitioner from one who optimises for appearances.

### Render Free Tier Constraints

The free tier's 512 MB RAM limit required careful memory management. A two-worker Gunicorn configuration would double the resident memory of the loaded model and Python runtime, exceeding the cap. Single-worker, two-thread mode was the correct trade-off: concurrent request handling is preserved without a second copy of the model in memory.

### Startup Reliability

Flask's default behaviour is to serve requests even when application-level resources are in a bad state (missing files, schema mismatches). Implementing fail-fast startup — calling `sys.exit(1)` before the server binds if any validation fails — required overriding this default. The tradeoff is zero tolerance for misconfigured deploys, which is the correct behaviour for a production service.

---

## Lessons Learned

### Data Leakage Is Easy to Introduce and Hard to Spot

Synthetic feature generation is a legitimate technique for extending small datasets. But any generated feature that references the target variable — even indirectly — introduces leakage. The corrected detection methodology (tracing generation code line by line, then computing correlation with `y`) is now a standard step in this project's training pipeline.

### Production ML Requires More Than a Trained Model

The model is one component. Startup validation, schema versioning, health endpoints, input sanitisation, security headers, and a deployment configuration are all required before a model can be called production-ready. This project implements all of them.

### Feature Importance Is a Post-Hoc Signal

A feature importance of 29.7% for `Claim_Amount` in v2.1 looked like the model had correctly identified a strong predictor. It was actually a leakage signal — the model was exploiting a shortcut, not learning a relationship. Feature importance alone does not validate a feature; its generation logic must be audited independently.

### Fail-Fast Is Friendlier Than Graceful Degradation

A server that starts in a degraded state and returns wrong predictions is worse than one that refuses to start at all. Fail-fast startup with explicit error messages makes misconfiguration failures loud, immediate, and actionable rather than silent and insidious.

### MLOps Basics Pay Off Immediately

Storing `model_metadata.json` alongside the pkl — with the feature list, training timestamp, and metrics — made the schema mismatch check trivially implementable. The thirty lines of validation code have already caught a real misconfiguration (the v2.1 → v2.2 feature schema change) exactly as designed.

---

## Future Improvements

- **Real extended features**: Replace synthetic `past_consultations`, `num_of_steps`, `NUmber_of_past_hospitalizations`, and `Anual_Salary` with data from a real healthcare dataset to further strengthen model validity
- **Hyperparameter tuning**: Add a `GridSearchCV` or `Optuna` sweep over the GBR parameter space with the clean 10-feature set
- **SHAP explanations**: Integrate SHAP values into the `/predict` response so users receive a per-feature contribution breakdown alongside the premium figure
- **Model versioning**: Add a model registry (MLflow or DVC) to track experiment history and enable rollback
- **CI/CD pipeline**: GitHub Actions workflow to run the training pipeline and unit tests on every push, blocking deploys on schema regression
- **Rate limiting**: Add `flask-limiter` to the `/predict` endpoint to prevent abuse on the public deployment
- **Persistent logging**: Route structured prediction logs to a time-series store (e.g. Logtail, Datadog) for drift monitoring over time

---

## Technologies Used

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| Web framework | Flask | 3.0.0 |
| ML library | Scikit-Learn | 1.6.1 |
| Numerical computing | NumPy | 1.26.4 |
| Model serialisation | Joblib | 1.3.2 |
| WSGI server | Gunicorn | 21.2.0 |
| Frontend | HTML5 / CSS3 / Vanilla JS | — |
| Deployment | Render | Free tier |

---

## Developer

**Arun VK**

Machine learning engineer and full-stack developer with a focus on production-quality ML systems.

- **Email**: [arunvk207@gmail.com](mailto:arunvk207@gmail.com)
- **LinkedIn**: [linkedin.com/in/arunvk2004](https://www.linkedin.com/in/arunvk2004/)

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Arun VK

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">Built with precision by <a href="https://www.linkedin.com/in/arunvk2004/">Arun VK</a></p>
