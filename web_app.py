# web_app.py (formerly app.py)
"""
Flask web app for House Price Prediction.

Requirements (install into your .venv):
    pip install flask pandas numpy joblib

Expectations:
 - A trained sklearn Pipeline saved as "model.pkl" in the same folder.
   Example to save after training:
     import joblib
     joblib.dump(pipeline, "model.pkl")

Usage:
  python web_app.py
  Open http://127.0.0.1:5000 in the browser
"""

from flask import Flask, request, render_template_string, redirect, url_for, flash
import joblib
import numpy as np
import pandas as pd
import os
import math

# Import configuration
from config import config

app = Flask(__name__)

# Load configuration based on environment variable or default to 'development'
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Get paths from config
MODEL_PATH = app.config['MODEL_PATH']

# --- Helper: convert price-like text (e.g. "42 Lac", "1.2 Cr", "15,00,000") to rupees ---
def convert_price_to_rupees(val):
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower().replace(",", "")
    if s == "":
        return np.nan
    if "lac" in s:
        # e.g. "42 lac" or "42lac"
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        try:
            return float(num) * 1e5
        except:
            return np.nan
    if "cr" in s or "crore" in s:
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        try:
            return float(num) * 1e7
        except:
            return np.nan
    # fallback: extract first numeric token
    import re
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    return np.nan

# --- Fields (must match features used during training) ---
NUMERIC_FIELDS = [
    "Amount(in rupees)",
    "Carpet Area",
    "Floor",
    "Bathroom",
    "Balcony",
    "Car Parking",
    "Super Area"
]

CATEGORICAL_FIELDS = [
    "location",
    "Status",
    "Transaction",
    "Furnishing",
    "facing",
    "overlooking",
    "Ownership"
]

ALL_FIELDS = NUMERIC_FIELDS + CATEGORICAL_FIELDS

# Try to load model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print(f"Model file '{MODEL_PATH}' not found. Train and save pipeline as 'model.pkl' then restart the app.")

# --- HTML template (inline) ---
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>House Price Prediction</title>
  <style>
    body {font-family: Arial, sans-serif; margin: 30px;}
    .container {max-width: 800px; margin: auto;}
    label {display:block; margin-top:10px;}
    input[type=text], select {width:100%; padding:8px; margin-top:4px; box-sizing: border-box;}
    .row {display:flex; gap: 12px;}
    .col {flex:1;}
    .submit-btn {margin-top:15px; padding:10px 16px; font-size:16px;}
    .result {padding:12px; margin-top:18px; background:#f4f4f4; border-radius:6px;}
    .note {font-size: 0.9em; color: #555;}
  </style>
</head>
<body>
  <div class="container">
    <h1>House Price Prediction</h1>
    <p class="note">Enter property details and click <b>Predict</b>. You can enter prices like "42 Lac" or "1.2 Cr".</p>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <ul>
          {% for msg in messages %}
            <li style="color: red">{{ msg }}</li>
          {% endfor %}
        </ul>
      {% endif %}
    {% endwith %}
    <form method="post" action="{{ url_for('predict') }}">
      <h3>Numeric inputs</h3>
      {% for f in numeric_fields %}
        <label>{{ f }}
          <input type="text" name="{{ f }}" value="{{ request.form.get(f, '') }}" placeholder="{% if 'Amount' in f %}e.g. 42 Lac or 1500000{% else %}enter number{% endif %}">
        </label>
      {% endfor %}
      <h3>Categorical inputs</h3>
      {% for f in categorical_fields %}
        <label>{{ f }}
          <input type="text" name="{{ f }}" value="{{ request.form.get(f, '') }}" placeholder="e.g. Thane, Ready to move, Resale">
        </label>
      {% endfor %}
      <button type="submit" class="submit-btn">Predict</button>
    </form>

    {% if prediction %}
      <div class="result">
        <h3>Prediction</h3>
        <p><b>Predicted Price (rupees):</b> {{ prediction | round(2) }}</p>
        <p><b>Predicted Price (formatted):</b> ₹ {{ formatted }}</p>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

# --- Utility: format rupee with commas ---
def rupee_format(x):
    try:
        # if numpy array, extract 0
        if np.ndim(x) > 0:
            x = float(x[0])
        x = float(x)
        # round to nearest rupee
        rounded = round(x)
        return f"{rounded:,}"
    except:
        return str(x)

# --- Route: index (form) ---
@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML,
                                  numeric_fields=NUMERIC_FIELDS,
                                  categorical_fields=CATEGORICAL_FIELDS,
                                  prediction=None,
                                  formatted=None)

# --- Route: predict (POST) ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not found. Train and save pipeline as 'model.pkl' before using the app.")
        return redirect(url_for("index"))

    # build input dict
    input_data = {}
    # Handle numeric fields
    for f in NUMERIC_FIELDS:
        raw = request.form.get(f, "").strip()
        if raw == "":
            # leave as NaN (model pipeline should handle imputation)
            input_data[f] = np.nan
        else:
            if f == "Amount(in rupees)":
                # convert textual rupee formats
                input_data[f] = convert_price_to_rupees(raw)
            else:
                # try numeric conversion
                try:
                    # if the user typed "2 Covered" or similar, extract number
                    import re
                    m = re.search(r"(\d+\.?\d*)", raw)
                    if m:
                        input_data[f] = float(m.group(1))
                    else:
                        input_data[f] = float(raw)
                except:
                    # fallback NaN
                    input_data[f] = np.nan

    # Handle categorical fields: keep as raw strings (pipeline will handle encoding & imputation)
    for f in CATEGORICAL_FIELDS:
        raw = request.form.get(f, "").strip()
        if raw == "":
            input_data[f] = np.nan
        else:
            input_data[f] = raw

    # create dataframe with single row and same column order
    # ensure columns exist exactly as during training:
    input_df = pd.DataFrame([input_data], columns=ALL_FIELDS)

    try:
        pred = model.predict(input_df)  # pipeline will preprocess then predict
    except Exception as e:
        flash(f"Prediction error: {e}")
        return redirect(url_for("index"))

    # If the pipeline was trained with log1p on target, and you saved the pipeline that outputs log1p predictions,
    # you must inverse-transform them here. This app can't know whether you log-transformed the target inside the pipeline.
    # Common approach: train pipeline to predict log1p(target) and then create a custom wrapper to inverse-transform.
    # If your pipeline already returns predictions in rupees, use directly.
    #
    # If your saved pipeline returns log1p predictions, uncomment:
    # pred = np.expm1(pred)

    pred_value = float(pred[0])

    # Format for display
    if pred_value >= 1e7:
        # show in Crores for readability
        formatted = f"{pred_value/1e7:.3f} Cr (₹ {rupee_format(pred_value)})"
    elif pred_value >= 1e5:
        formatted = f"{pred_value/1e5:.3f} Lac (₹ {rupee_format(pred_value)})"
    else:
        formatted = f"₹ {rupee_format(pred_value)}"

    return render_template_string(INDEX_HTML,
                                  numeric_fields=NUMERIC_FIELDS,
                                  categorical_fields=CATEGORICAL_FIELDS,
                                  prediction=pred_value,
                                  formatted=formatted,
                                  request=request)

# --- Run app ---
if __name__ == "__main__":
    # Use debug mode from config
    debug_mode = app.config.get('DEBUG', False)
    app.run(debug=debug_mode)
