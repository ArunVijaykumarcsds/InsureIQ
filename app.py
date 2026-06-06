"""
InsureIQ — Production Flask application
Render Free Tier optimised: single worker, fail-fast startup, full model + metadata validation.
"""
import os
import sys
import json
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
VERSION    = '2.2.0'
START_TIME = time.time()

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, 'insurance_model.pkl')
METADATA_PATH = os.path.join(BASE_DIR, 'model_metadata.json')

FEATURE_ORDER = [
    'age', 'sex', 'bmi', 'children', 'smoker',
    'past_consultations', 'num_of_steps',
    'NUmber_of_past_hospitalizations',
    'Anual_Salary', 'region',
]

REQUIRED_FIELDS = [
    'age', 'sex', 'bmi', 'children', 'smoker',
    'past_consultations', 'num_of_steps',
    'past_hospitalizations',
    'annual_salary', 'region',
]


# ── Startup validation ─────────────────────────────────────────────────────────
def _abort(msg: str):
    log.error("[ERROR] %s", msg)
    sys.exit(1)


def load_model():
    """Load model and validate it can produce predictions. Fail-fast on any issue."""
    if not os.path.exists(MODEL_PATH):
        _abort(f"insurance_model.pkl not found at: {MODEL_PATH}\n"
               "  → Run: python train_model.py")

    try:
        m = joblib.load(MODEL_PATH)
    except Exception as exc:
        _abort(f"Failed to deserialise model: {exc}")

    # Smoke-test with a zero-vector
    try:
        dummy = np.zeros((1, len(FEATURE_ORDER)))
        result = m.predict(dummy)
        assert len(result) == 1, "predict() returned unexpected shape"
    except Exception as exc:
        _abort(f"Model smoke-test failed: {exc}")

    size_kb = os.path.getsize(MODEL_PATH) // 1024
    log.info("[OK] Model loaded (%d KB)", size_kb)
    return m


def load_metadata() -> dict:
    """Load and validate model_metadata.json. Fail-fast if missing or malformed."""
    if not os.path.exists(METADATA_PATH):
        _abort(f"model_metadata.json not found at: {METADATA_PATH}\n"
               "  → Run: python train_model.py")

    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
    except Exception as exc:
        _abort(f"Failed to parse model_metadata.json: {exc}")

    # Validate feature schema matches app's expected features
    stored_features = meta.get('features', [])
    if stored_features != FEATURE_ORDER:
        _abort(
            f"Feature schema mismatch!\n"
            f"  Metadata: {stored_features}\n"
            f"  App expects: {FEATURE_ORDER}\n"
            "  → Re-run train_model.py to regenerate a compatible model."
        )

    log.info("[OK] Metadata loaded — trained %s, dataset_size=%d, R²=%.4f, MAE=$%.2f",
             meta.get('trained_at', 'unknown')[:10],
             meta.get('dataset_size', 0),
             meta.get('metrics', {}).get('r2', 0),
             meta.get('metrics', {}).get('mae', 0))
    return meta


# ── Load at module import (fail-fast before Flask binds) ──────────────────────
model    = load_model()
metadata = load_metadata()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
log.info("[OK] Flask initialised")


# ── Security headers ───────────────────────────────────────────────────────────
@app.after_request
def security_headers(response):
    h = response.headers
    h['X-Content-Type-Options']  = 'nosniff'
    h['X-Frame-Options']         = 'DENY'
    h['X-XSS-Protection']        = '1; mode=block'
    h['Referrer-Policy']         = 'strict-origin-when-cross-origin'
    h['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "connect-src 'self';"
    )
    return response


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify(error="Bad request — check your input."), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Resource not found."), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify(error="Method not allowed."), 405

@app.errorhandler(500)
def server_error(e):
    log.exception("Unhandled 500: %s", e)
    return jsonify(error="An internal error occurred. Please try again."), 500


# ── Helpers ────────────────────────────────────────────────────────────────────
def sanitise_float(value, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number, got: {value!r}")


def predict_premium(inputs: dict) -> float:
    features = [inputs[f] for f in FEATURE_ORDER]
    arr = np.array(features, dtype=float).reshape(1, -1)
    return max(0.0, round(float(model.predict(arr)[0]), 2))


def get_risk_profile(prediction: float, age: float, bmi: float, smoker: float) -> dict:
    score = 0
    if smoker == 1:          score += 40
    if bmi >= 30:            score += 20
    elif bmi >= 25:          score += 10
    if age >= 55:            score += 20
    elif age >= 40:          score += 10
    if prediction > 20000:   score += 20
    elif prediction > 10000: score += 10

    if score >= 60:
        return dict(level="High Risk", color="high", icon="🔴", score=score,
                    advice="Consider lifestyle changes. Smoking cessation and BMI reduction can significantly lower your premium.")
    elif score >= 30:
        return dict(level="Medium Risk", color="medium", icon="🟡", score=score,
                    advice="Moderate risk profile. Regular checkups and a healthy BMI can help reduce costs.")
    else:
        return dict(level="Low Risk", color="low", icon="🟢", score=score,
                    advice="Great profile! You're in a low-risk category. Maintain your healthy habits to keep premiums low.")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', version=VERSION)


@app.route('/predict', methods=['POST'])
def predict():
    raw = request.get_json(silent=True)
    if not raw:
        return jsonify(error="Request body must be JSON."), 400

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        return jsonify(error=f"Missing required fields: {', '.join(missing)}"), 400

    try:
        inputs = {
            'age':                             sanitise_float(raw['age'],                  'age'),
            'sex':                             sanitise_float(raw['sex'],                  'sex'),
            'bmi':                             sanitise_float(raw['bmi'],                  'bmi'),
            'children':                        sanitise_float(raw['children'],             'children'),
            'smoker':                          sanitise_float(raw['smoker'],               'smoker'),
            'past_consultations':              sanitise_float(raw['past_consultations'],   'past_consultations'),
            'num_of_steps':                    sanitise_float(raw['num_of_steps'],         'num_of_steps'),
            'NUmber_of_past_hospitalizations': sanitise_float(raw['past_hospitalizations'],'past_hospitalizations'),
            'Anual_Salary':                    sanitise_float(raw['annual_salary'],        'annual_salary'),
            'region':                          sanitise_float(raw['region'],               'region'),
        }
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    validations = [
        (0 <= inputs['age'] <= 120,     "age must be 0–120"),
        (0 <= inputs['bmi'] <= 80,      "BMI must be 0–80"),
        (inputs['sex'] in (0, 1),       "sex must be 0 or 1"),
        (inputs['smoker'] in (0, 1),    "smoker must be 0 or 1"),
        (0 <= inputs['children'] <= 10, "children must be 0–10"),
        (0 <= inputs['region'] <= 3,    "region must be 0–3"),
    ]
    for ok, msg in validations:
        if not ok:
            return jsonify(error=f"Validation error: {msg}"), 400

    try:
        prediction = predict_premium(inputs)
        risk       = get_risk_profile(prediction, inputs['age'], inputs['bmi'], inputs['smoker'])

        whatif_nonsmoker = predict_premium({**inputs, 'smoker': 0})       if inputs['smoker'] == 1   else None
        whatif_bmi       = predict_premium({**inputs, 'bmi': 24.9})       if inputs['bmi'] > 24.9    else None
        whatif_age       = predict_premium({**inputs, 'age': inputs['age'] - 10}) if inputs['age'] > 30 else None

        log.info("Prediction — age=%.0f bmi=%.1f smoker=%.0f → $%.2f (%s)",
                 inputs['age'], inputs['bmi'], inputs['smoker'], prediction, risk['level'])

        return jsonify(
            prediction=prediction,
            risk=risk,
            whatif=dict(
                nonsmoker=whatif_nonsmoker,
                healthy_bmi=whatif_bmi,
                age_diff=whatif_age,
                current_age=inputs['age'],
                current_bmi=inputs['bmi'],
                is_smoker=inputs['smoker'] == 1,
            )
        )
    except Exception:
        log.exception("Prediction pipeline error")
        return jsonify(error="Prediction failed. Please try again."), 500


@app.route('/health')
def health():
    return jsonify(status="healthy", model_loaded=True, version=VERSION), 200


@app.route('/system-status')
def system_status():
    try:
        import psutil
        mem_mb = round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    except ImportError:
        mem_mb = None

    m = metadata.get('metrics', {})
    return jsonify(
        status           = "healthy",
        version          = VERSION,
        uptime_seconds   = int(time.time() - START_TIME),
        model_loaded     = True,
        model_path       = MODEL_PATH,
        model_size_kb    = os.path.getsize(MODEL_PATH) // 1024,
        deployment_mode  = os.environ.get('FLASK_ENV', 'production'),
        memory_mb        = mem_mb,
        model_metadata   = dict(
            trained_at     = metadata.get('trained_at', '')[:19],
            dataset_source = metadata.get('dataset_source', ''),
            dataset_size   = metadata.get('dataset_size', 0),
            algorithm      = metadata.get('algorithm', ''),
            feature_count  = metadata.get('feature_count', 0),
            metrics        = dict(
                mae        = m.get('mae'),
                rmse       = m.get('rmse'),
                r2         = m.get('r2'),
                cv_r2_mean = m.get('cv_r2_mean'),
                cv_r2_std  = m.get('cv_r2_std'),
            ),
            top_features = sorted(
                m.get('feature_importances', {}).items(),
                key=lambda x: -x[1]
            )[:5],
        ),
    ), 200


log.info("[OK] Prediction pipeline ready — InsureIQ v%s", VERSION)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
