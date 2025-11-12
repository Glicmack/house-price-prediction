# api.py (formerly app_api.py)
"""
Flask app that serves:
 - GET /       -> simple HTML form (optional)
 - POST /predict -> form-based POST (optional)
 - POST /predict_api -> JSON API, expects {"feature1": value, ...}
 
Usage:
    (.venv) python api.py
    curl example (see below)
"""

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, flash
import joblib
import json
import numpy as np
import pandas as pd
import os

# Import configuration
from config import config

app = Flask(__name__)

# Load configuration based on environment variable or default to 'development'
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Get paths from config
MODEL_PATH = app.config['MODEL_PATH']
META_PATH = app.config['META_PATH']

# Load model & metadata
model = None
meta = None
if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("Model and metadata loaded.")
else:
    print("Model or metadata not found. Run save_model.py first to create them.")

# Helper price converter (same as training)
def convert_price_to_rupees(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower().replace(",", "")
    if s == "":
        return np.nan
    if "lac" in s:
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        return float(num) * 1e5
    if "cr" in s or "crore" in s:
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        return float(num) * 1e7
    import re
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return np.nan

def rupee_format(x):
    try:
        x = float(x)
        rounded = round(x)
        return f"{rounded:,}"
    except:
        return str(x)

# Simple web form (optional)
INDEX_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>House Price API</title></head>
  <body>
    <h2>House Price Prediction API</h2>
    <p>Use POST /predict_api with JSON to get predictions. Example using curl is below.</p>
    <p>curl example:</p>
    <pre>
curl -X POST http://127.0.0.1:5000/predict_api \
 -H "Content-Type: application/json" \
 -d '{"Amount(in rupees)": "42 Lac", "Carpet Area": 630, "Floor": 3, "Bathroom": 2, "Balcony": 1, "Car Parking": 1, "Super Area": 950, "location":"Thane", "Status":"Ready to move", "Transaction":"Resale", "Furnishing":"Semi-Furnished", "facing":"North", "overlooking":"Park", "Ownership":"Freehold"}'
    </pre>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

# Helper: build input DataFrame (single-row) from dict
def build_input_df(payload):
    if meta is None:
        raise RuntimeError("Metadata not loaded")

    features = meta["all_features"]
    row = {}
    for f in features:
        # prefer provided value or NaN
        if f not in payload:
            row[f] = np.nan
        else:
            v = payload[f]
            # if this is the Amount field, convert textual rupee formats
            if f == "Amount(in rupees)":
                row[f] = convert_price_to_rupees(v)
            else:
                # for numeric columns – try to parse number from strings like '2 Open'
                if f in meta["numeric_features"]:
                    if pd.isna(v):
                        row[f] = np.nan
                    else:
                        try:
                            import re
                            if isinstance(v, str):
                                m = re.search(r"(\d+\.?\d*)", v)
                                if m:
                                    row[f] = float(m.group(1))
                                else:
                                    row[f] = float(v)
                            else:
                                row[f] = float(v)
                        except:
                            row[f] = np.nan
                else:
                    # categorical – keep as string
                    if v is None:
                        row[f] = np.nan
                    else:
                        row[f] = str(v)
    df = pd.DataFrame([row], columns=features)
    return df

@app.route("/predict_api", methods=["POST"])
def predict_api():
    if model is None or meta is None:
        return jsonify({"error": "Model not loaded. Run training first."}), 500

    # Expect JSON payload
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "detail": str(e)}), 400

    # Build input df
    try:
        input_df = build_input_df(payload)
    except Exception as e:
        return jsonify({"error": "Failed to build input", "detail": str(e)}), 400

    # Predict
    try:
        pred = model.predict(input_df)
    except Exception as e:
        return jsonify({"error": "Prediction error", "detail": str(e)}), 500

    pred_val = float(pred[0])

    # inverse log if needed
    if meta.get("log_target", False):
        pred_val = float(np.expm1(pred_val))

    # formatted
    if pred_val >= 1e7:
        formatted = f"{pred_val/1e7:.3f} Cr (₹ {rupee_format(pred_val)})"
    elif pred_val >= 1e5:
        formatted = f"{pred_val/1e5:.3f} Lac (₹ {rupee_format(pred_val)})"
    else:
        formatted = f"₹ {rupee_format(pred_val)}"

    response = {
        "pred_rupees": pred_val,
        "pred_formatted": formatted,
        "input_received": payload
    }
    return jsonify(response), 200

# Optional form-based endpoint that forwards to predict_api
@app.route("/predict", methods=["POST"])
def predict_form():
    # build payload from form fields and redirect to /predict_api to reuse logic
    if meta is None:
        flash("Model not ready. Train the model first.")
        return redirect(url_for("index"))

    payload = {}
    for f in meta["all_features"]:
        payload[f] = request.form.get(f)

    # forward to predict_api logic
    from flask import Response
    with app.test_request_context(json=payload):
        resp = predict_api()
        # resp is (jsonify(...), status_code)
        if isinstance(resp, tuple):
            body, code = resp
            return body, code
        return resp

if __name__ == "__main__":
    # Use debug mode from config
    debug_mode = app.config.get('DEBUG', False)
    app.run(debug=debug_mode)
