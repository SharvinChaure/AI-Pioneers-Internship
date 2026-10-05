"""Flask prediction API + web UI for the Breast Cancer Diagnosis model."""
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
model = joblib.load("models/model.joblib")
with open("models/metadata.json") as f:
    META = json.load(f)
FEATURES = META["features"]
LABELS = META["target_names"]  # index 0 = malignant, 1 = benign


def _parse(payload):
    """Validate payload and return a (1, n_features) array."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    try:
        values = [float(payload[f]) for f in FEATURES]
    except (TypeError, ValueError):
        raise ValueError("All feature values must be numeric.")
    if any(v < 0 or not np.isfinite(v) for v in values):
        raise ValueError("Feature values must be finite and non-negative.")
    return pd.DataFrame([values], columns=FEATURES)


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, ranges=META["feature_ranges"], metrics=META["metrics"])


@app.route("/health")
def health():
    return jsonify(status="ok", model=META["metrics"]["model"])


@app.route("/features")
def features():
    return jsonify(features=FEATURES, ranges=META["feature_ranges"])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    try:
        x = _parse(payload)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    proba = model.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    return jsonify(prediction=LABELS[idx], class_index=idx,
                   confidence=round(float(proba[idx]), 4),
                   probabilities={LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
