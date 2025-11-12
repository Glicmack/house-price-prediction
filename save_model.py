# save_model.py
"""
Train a pipeline on house_prices.csv and save model.pkl + metadata.json.

Usage:
    (.venv) python save_model.py
Outputs:
    - model.pkl        (joblib file)
    - metadata.json    (contains feature names & log_target flag)
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------- CONFIG -----------------
CSV_PATH = "house_prices.csv"
MODEL_PATH = "model.pkl"
META_PATH = "metadata.json"
LOG_TARGET = True   # set False if you don't want log1p on target
RANDOM_STATE = 42
# ------------------------------------------

# Helper converters (same logic as used in app)
def convert_price_to_rupees(val):
    if pd.isna(val):
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

def extract_first_number(series):
    return pd.to_numeric(series.astype(str).str.extract(r"(\d+\.?\d*)")[0], errors='coerce')

# Load CSV
df = pd.read_csv(CSV_PATH)

# Drop long-text columns we aren't using in baseline model
df = df.drop(columns=["Index", "Title", "Description", "Society", "Dimensions", "Plot Area"], errors='ignore')
df.columns = [c.strip() for c in df.columns]

# Convert price-like columns
if "Amount(in rupees)" in df.columns:
    df["Amount(in rupees)"] = df["Amount(in rupees)"].apply(convert_price_to_rupees)

if "Price (in rupees)" in df.columns and df["Price (in rupees)"].dtype == object:
    df["Price (in rupees)"] = df["Price (in rupees)"].apply(convert_price_to_rupees)

# Extract numeric parts for certain columns
for col in ["Bathroom", "Balcony", "Car Parking", "Floor", "Super Area", "Carpet Area", "Plot Area"]:
    if col in df.columns:
        df[col] = extract_first_number(df[col])

# Fill medians for numeric columns and mode for object columns (simple, consistent)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode().iloc[0])

# Confirm target exists
target_col = "Price (in rupees)"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' missing in CSV")

# Prepare X, y
X = df.drop(columns=[target_col])
y = df[target_col].copy()

# optional log transform
if LOG_TARGET:
    y = np.log1p(y)

# Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Build preprocessing and pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
], remainder="drop")  # drop any other columns not specified

regressor = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", regressor)
])

# Train-test split (optional local eval)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

print("Training pipeline... (this may take a while)")
pipeline.fit(X_train, y_train)
print("Training complete")

# Evaluate quickly (inverse transform if log target)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
y_pred = pipeline.predict(X_test)
if LOG_TARGET:
    y_test_inv = np.expm1(y_test)
    y_pred_inv = np.expm1(y_pred)
else:
    y_test_inv = y_test
    y_pred_inv = y_pred

print("R2:", round(r2_score(y_test_inv, y_pred_inv), 4))
print("MAE:", round(mean_absolute_error(y_test_inv, y_pred_inv), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)), 2))

# Save model
joblib.dump(pipeline, MODEL_PATH)
print(f"Saved pipeline to {MODEL_PATH}")

# Write metadata
metadata = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "all_features": numeric_features + categorical_features,
    "log_target": bool(LOG_TARGET),
    "target_col": target_col
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {META_PATH}")
