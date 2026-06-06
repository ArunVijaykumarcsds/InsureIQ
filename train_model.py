"""
InsureIQ — Production ML Training Pipeline
==========================================
Strategy:
  - Downloads the Medical Cost Personal Dataset (1,338 real insurance records)
    published by Brett Lantz in "Machine Learning with R" (widely used benchmark).
  - The real dataset covers: age, sex, bmi, children, smoker, region, charges.
  - Extended features (past_consultations, num_of_steps,
    NUmber_of_past_hospitalizations, Anual_Salary)
    are generated with realistic statistical correlations anchored to each
    individual's real risk profile — preserving the true premium signal.
  - A full sklearn Pipeline (scaling + GradientBoostingRegressor) is trained,
    evaluated (MAE, RMSE, R²), and exported with metadata.

Usage:
    python train_model.py              # downloads dataset, trains, exports
    python train_model.py --offline    # skip download, use local cache only
"""
import os
import sys
import json
import logging
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'insurance_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')
CACHE_PATH    = os.path.join(BASE_DIR, 'insurance_base.csv')

DATASET_URL = (
    'https://raw.githubusercontent.com/stedy/'
    'Machine-Learning-with-R-datasets/master/insurance.csv'
)

# Matches FEATURE_ORDER in app.py exactly — 10 clean features, no leakage
FEATURE_ORDER = [
    'age', 'sex', 'bmi', 'children', 'smoker',
    'past_consultations', 'num_of_steps',
    'NUmber_of_past_hospitalizations',
    'Anual_Salary', 'region',
]

VERSION = '2.2.0'


# ── Dataset acquisition ────────────────────────────────────────────────────────
def fetch_dataset() -> list[dict]:
    """Download (or load from cache) the Medical Cost Personal Dataset."""
    if os.path.exists(CACHE_PATH):
        log.info("Using cached base dataset: %s", CACHE_PATH)
        raw = open(CACHE_PATH).read()
    else:
        import urllib.request
        log.info("Downloading Medical Cost Personal Dataset ...")
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=15) as r:
                raw = r.read().decode()
            with open(CACHE_PATH, 'w') as f:
                f.write(raw)
            log.info("  -> Saved to cache: %s", CACHE_PATH)
        except Exception as exc:
            log.error("Download failed: %s", exc)
            log.error("Ensure internet access or pre-place insurance_base.csv in the project root.")
            sys.exit(1)

    lines = raw.strip().split('\n')
    header = lines[0].split(',')
    rows = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))

    log.info("Base dataset: %d records, columns: %s", len(rows), header)
    return rows


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(rows: list[dict], rng) -> tuple:
    """
    Extend the 7-column real dataset to the 10-column app schema.

    Extended features are generated with medically plausible correlations:
      past_consultations              — higher for older, heavier, sicker individuals
      num_of_steps                    — lower for smokers, obese; higher for young & healthy
      NUmber_of_past_hospitalizations — driven by smoker, bmi, age
      Anual_Salary                    — weakly correlated with age (experience proxy)

    NOTE: The two previously-leaked features (past claim amount and hospital
    expenditure) have been permanently removed. Both were derived from the
    training target and constituted data leakage.
    """
    import numpy as np

    n = len(rows)
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

    age      = np.array([int(r['age'])                              for r in rows], dtype=float)
    sex      = np.array([0 if r['sex'] == 'female' else 1          for r in rows], dtype=float)
    bmi      = np.array([float(r['bmi'])                           for r in rows], dtype=float)
    children = np.array([int(r['children'])                        for r in rows], dtype=float)
    smoker   = np.array([1 if r['smoker'] == 'yes' else 0          for r in rows], dtype=float)
    region   = np.array([region_map.get(r['region'], 0)            for r in rows], dtype=float)
    charges  = np.array([float(r['charges'])                       for r in rows], dtype=float)

    # ── Past consultations (0–20)
    consult_base = (
        (age - 18) / (64 - 18) * 8
        + smoker * 3.5
        + np.clip((bmi - 25) / 10, 0, 3)
        + children * 0.5
    )
    past_consultations = np.clip(
        rng.poisson(np.clip(consult_base, 0.5, 15)),
        0, 20
    ).astype(float)

    # ── Daily steps (0–20 000)
    steps_mean = (
        10000
        - smoker * 2500
        - np.clip((bmi - 22) * 150, 0, 3000)
        - (age - 18) * 50
        + rng.normal(0, 500, n)
    )
    num_of_steps = np.clip(
        rng.normal(steps_mean, 1200),
        0, 20000
    ).astype(float)

    # ── Past hospitalisations (0–10)
    hosp_base = (
        smoker * 2.2
        + np.clip((bmi - 30) / 10, 0, 2)
        + (age - 18) / (64 - 18) * 3
    )
    NUmber_of_past_hospitalizations = np.clip(
        rng.poisson(np.clip(hosp_base, 0.1, 8)),
        0, 10
    ).astype(float)

    # ── Annual salary — weak positive correlation with age (career progression)
    salary_base = (
        30000
        + np.clip((age - 22) * 1200, 0, 120000)
        + sex * rng.normal(3000, 2000, n)
        + rng.normal(0, 12000, n)
    )
    Anual_Salary = np.clip(
        np.round(salary_base / 1000) * 1000,
        20000, 300000
    ).astype(float)

    X = np.column_stack([
        age, sex, bmi, children, smoker,
        past_consultations, num_of_steps,
        NUmber_of_past_hospitalizations,
        Anual_Salary, region,
    ])
    y = charges.copy()

    return X, y, {
        'age': age, 'smoker': smoker, 'bmi': bmi,
        'charges': charges, 'past_consultations': past_consultations,
        'num_of_steps': num_of_steps,
    }


# ── Model training & evaluation ────────────────────────────────────────────────
def train_and_evaluate(X, y) -> tuple:
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    log.info("Splitting data: 80%% train / 20%% test (stratified on smoker column) ...")
    smoker_col = X[:, 4].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=smoker_col
    )
    log.info("  Train: %d  Test: %d", len(X_tr), len(X_te))

    log.info("Building GradientBoostingRegressor pipeline ...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('gbr', GradientBoostingRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.85,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            warm_start=False,
        )),
    ])

    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 1)
    log.info("  Training complete in %.1f s", elapsed)

    # ── Hold-out metrics
    y_pred = pipe.predict(X_te)
    mae    = round(mean_absolute_error(y_te, y_pred), 2)
    rmse   = round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 2)
    r2     = round(float(r2_score(y_te, y_pred)), 4)

    log.info("── Hold-out Evaluation ──────────────────────────────")
    log.info("  MAE   : $%.2f",  mae)
    log.info("  RMSE  : $%.2f",  rmse)
    log.info("  R2    : %.4f",   r2)
    log.info("────────────────────────────────────────────────────")

    # ── 5-fold CV R² on full dataset
    cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2')
    cv_r2_mean = round(float(cv_r2.mean()), 4)
    cv_r2_std  = round(float(cv_r2.std()),  4)
    log.info("  5-fold CV R2: %.4f +/- %.4f", cv_r2_mean, cv_r2_std)

    # ── Feature importance (from GBR)
    gbr = pipe.named_steps['gbr']
    importances = dict(zip(FEATURE_ORDER, [round(float(v), 4) for v in gbr.feature_importances_]))
    top = sorted(importances.items(), key=lambda x: -x[1])[:5]
    log.info("  Top features: %s", top)

    metrics = {
        'mae':        mae,
        'rmse':       rmse,
        'r2':         r2,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std':  cv_r2_std,
        'train_size': int(len(X_tr)),
        'test_size':  int(len(X_te)),
        'feature_importances': importances,
        'training_seconds': elapsed,
    }
    return pipe, metrics


# ── Metadata persistence ───────────────────────────────────────────────────────
def save_metadata(metrics: dict, dataset_size: int):
    from datetime import datetime, timezone

    meta = {
        'version':        VERSION,
        'trained_at':     datetime.now(timezone.utc).isoformat(),
        'dataset_source': DATASET_URL,
        'dataset_size':   dataset_size,
        'features':       FEATURE_ORDER,
        'feature_count':  len(FEATURE_ORDER),
        'target':         'annual_insurance_charges_usd',
        'algorithm':      'GradientBoostingRegressor (sklearn Pipeline + StandardScaler)',
        'metrics':        metrics,
        'model_path':     MODEL_PATH,
        'feature_schema': {
            'age':                            {'type': 'float',  'range': [18, 64]},
            'sex':                            {'type': 'binary', 'values': [0, 1]},
            'bmi':                            {'type': 'float',  'range': [15.0, 55.0]},
            'children':                       {'type': 'int',    'range': [0, 5]},
            'smoker':                         {'type': 'binary', 'values': [0, 1]},
            'past_consultations':             {'type': 'int',    'range': [0, 20]},
            'num_of_steps':                   {'type': 'float',  'range': [0, 20000]},
            'NUmber_of_past_hospitalizations':{'type': 'int',    'range': [0, 10]},
            'Anual_Salary':                   {'type': 'float',  'range': [20000, 300000]},
            'region':                         {'type': 'int',    'range': [0, 3]},
        },
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    log.info("[OK] Metadata saved -> %s", METADATA_PATH)
    return meta


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='InsureIQ model training pipeline')
    parser.add_argument('--offline', action='store_true',
                        help='Skip download, use cached insurance_base.csv only')
    args = parser.parse_args()

    import numpy as np
    import joblib

    log.info("InsureIQ ML Training Pipeline v%s", VERSION)

    if args.offline and not os.path.exists(CACHE_PATH):
        log.error("--offline specified but no cache found at %s", CACHE_PATH)
        sys.exit(1)
    rows = fetch_dataset()

    rng = np.random.default_rng(seed=42)
    X, y, debug = engineer_features(rows, rng)
    log.info("Feature matrix: %s  |  target range: $%.0f - $%.0f",
             X.shape, y.min(), y.max())

    pipeline, metrics = train_and_evaluate(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    size_kb = os.path.getsize(MODEL_PATH) // 1024
    log.info("[OK] Model saved -> %s (%d KB)", MODEL_PATH, size_kb)

    save_metadata(metrics, dataset_size=len(rows))

    log.info("Training complete")
    log.info("  MAE $%.2f  |  RMSE $%.2f  |  R2 %.4f",
             metrics['mae'], metrics['rmse'], metrics['r2'])


if __name__ == '__main__':
    main()
