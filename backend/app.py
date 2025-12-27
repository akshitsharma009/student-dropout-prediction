from flask import Flask, request, jsonify
import joblib

from utils.preprocess import preprocess_input

app = Flask(__name__)

# Load trained model
model = joblib.load("../models/dropout_model.pkl")

# Business threshold
THRESHOLD = 0.6


def get_risk_bucket(probability):
    if probability < 0.4:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


@app.route("/", methods=["GET"])
def home():
    return "Student Dropout Prediction API running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if data is None:
            return jsonify({"error": "No JSON data provided"}), 400

        # Preprocess input
        X = preprocess_input(data)

        # Probability of Dropout (class = 1)
        proba = model.predict_proba(X)[0][1]

        # Decision based on threshold
        prediction = "Dropout" if proba >= THRESHOLD else "No Dropout"

        # Risk bucket
        risk_level = get_risk_bucket(proba)

        return jsonify({
            "prediction": prediction,
            "risk_level": risk_level,
            "probability": round(float(proba), 3),
            "threshold_used": THRESHOLD
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
