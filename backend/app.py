"""
app.py -- Flask REST API for the Student Academic Risk Prediction system.

Exposes two prediction modes, matching the two deployable models:
  - "screener"       : no grades required, usable from day one of term.
  - "early_warning"   : requires G1 and G2 (first two period grades),
                        substantially higher accuracy.

Both thresholds are imported from utils/config.py -- the SAME constants
used during threshold selection in the training notebook, so the deployed
decision boundary can never silently drift from what was actually
validated (this is a direct fix for a bug in the original version, where
the notebook validated a 0.35 threshold but the deployed API used a
hardcoded 0.6, undoing the entire point of the threshold-tuning work).
"""

from flask import Flask, request, jsonify
import joblib

from utils.config import (
    THRESHOLD_SCREENER,
    THRESHOLD_EARLY_WARNING,
    MODEL_PATH_SCREENER,
    MODEL_PATH_EARLY_WARNING,
    NUMERIC_FEATURES_SCREENER,
    NUMERIC_FEATURES_EARLY_WARNING,
    get_risk_bucket,
)
from utils.preprocess import build_feature_row

app = Flask(__name__)

screener_model = joblib.load(f"../models/{MODEL_PATH_SCREENER}")
early_warning_model = joblib.load(f"../models/{MODEL_PATH_EARLY_WARNING}")

MODES = {
    "screener": {
        "model": screener_model,
        "threshold": THRESHOLD_SCREENER,
        "numeric_features": NUMERIC_FEATURES_SCREENER,
    },
    "early_warning": {
        "model": early_warning_model,
        "threshold": THRESHOLD_EARLY_WARNING,
        "numeric_features": NUMERIC_FEATURES_EARLY_WARNING,
    },
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Student Academic Risk Prediction API running",
        "modes": list(MODES.keys()),
        "usage": "POST /predict with {\"mode\": \"screener\" | \"early_warning\", ...features}",
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        mode = data.get("mode", "screener")
        if mode not in MODES:
            return jsonify({"error": f"mode must be one of {list(MODES.keys())}"}), 400

        if mode == "early_warning" and ("G1" not in data or "G2" not in data):
            return jsonify({
                "error": "early_warning mode requires G1 and G2 (first and second period grades)"
            }), 400

        config = MODES[mode]
        X = build_feature_row(data, config["numeric_features"])

        proba = config["model"].predict_proba(X)[0][1]
        threshold = config["threshold"]
        prediction = "At Risk" if proba >= threshold else "Not At Risk"
        risk_level = get_risk_bucket(proba)

        return jsonify({
            "mode": mode,
            "prediction": prediction,
            "risk_level": risk_level,
            "probability": round(float(proba), 3),
            "threshold_used": threshold,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
