# gpt_improved.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# optional: XGBoost if installed
try:
    from xgboost import XGBRegressor
    xgb_available = True
except Exception:
    xgb_available = False

# ---- Load ----
df = pd.read_csv("house_prices.csv")

# ---- Drop obvious text-heavy columns that we won't use directly ----
df = df.drop(columns=["Index", "Title", "Description", "Society", "Dimensions", "Plot Area"], errors='ignore')

# ---- Helper converters ----
def convert_price_to_rupees(val):
    # Handles values like '42 Lac', '1.2 Cr', '15,00,000', or numeric strings
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower().replace(",", "")
    # common formats: '42 lac', '1.2 cr', '1500000'
    if "lac" in s:
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        return float(num) * 1e5
    if "cr" in s or "crore" in s:
        num = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
        return float(num) * 1e7
    # try extract first numeric token
    import re
    m = re.search(r"(\d+\.?\d*)", s)
    if m:
        return float(m.group(1))
    return np.nan

def extract_numeric_from_text(series):
    # returns float series with numeric extraction
    return series.astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)

# ---- Clean numeric-ish columns ----
# Normalize column names (optional)
df.columns = [c.strip() for c in df.columns]

# Convert price-like columns
if "Amount(in rupees)" in df.columns:
    df["Amount(in rupees)"] = df["Amount(in rupees)"].apply(convert_price_to_rupees)

# Price (target) might already be numeric or not
if "Price (in rupees)" in df.columns and df["Price (in rupees)"].dtype == object:
    df["Price (in rupees)"] = df["Price (in rupees)"].apply(convert_price_to_rupees)

# Clean integer-count columns that are stored as text
for col in ["Bathroom", "Balcony", "Car Parking", "Floor", "Super Area", "Carpet Area", "Plot Area"]:
    if col in df.columns:
        # For numeric-like text, extract first number
        ser = df[col].astype(str)
        # If column contains things like '1 Covered' or '2 Open', extract digits
        nums = ser.str.extract(r"(\d+\.?\d*)")[0]
        # convert to float and assign back (use explicit assignment to avoid warnings)
        df[col] = pd.to_numeric(nums, errors='coerce')

# ---- Now handle missing values WITHOUT chained assignment ----
# numeric median fill
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# categorical fill (mode)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)

# ---- Quick leakage check: correlation between Amount and Price ----
if "Amount(in rupees)" in df.columns and "Price (in rupees)" in df.columns:
    corr = df["Amount(in rupees)"].corr(df["Price (in rupees)"])
    print(f"Correlation between Amount(in rupees) and Price (target): {corr:.4f}")
    if corr > 0.9:
        print("WARNING: 'Amount(in rupees)' is highly correlated with target. It may be target leakage.")
        # Optionally drop it below:
        # df = df.drop(columns=["Amount(in rupees)"])

# ---- Prepare features and target ----
target_col = "Price (in rupees)"
if target_col not in df.columns:
    raise KeyError(f"Target column '{target_col}' not found in dataframe.")

X = df.drop(columns=[target_col])
y = df[target_col].copy()

# Option: log-transform target to stabilize variance (uncomment if desired)
log_target = True
if log_target:
    # avoid log(0)
    y = np.log1p(y)

# ---- Identify numeric vs categorical columns for pipeline ----
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---- Preprocessing pipeline ----
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ---- Model Pipeline ----
rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', rf)])

# ---- Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training model...")
pipeline.fit(X_train, y_train)

# ---- Predict and evaluate ----
y_pred = pipeline.predict(X_test)
if log_target:
    # inverse transform
    y_test_inv = np.expm1(y_test)
    y_pred_inv = np.expm1(y_pred)
else:
    y_test_inv = y_test
    y_pred_inv = y_pred

print("R2:", round(r2_score(y_test_inv, y_pred_inv), 4))
print("MAE:", round(mean_absolute_error(y_test_inv, y_pred_inv), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)), 2))

# ---- Feature importances (for tree models) ----
# Get feature names after onehot encoding
ohe_cols = []
if categorical_features:
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    # build full categorical feature names
    cat_names = ohe.get_feature_names_out(categorical_features).tolist()
    ohe_cols = cat_names

feature_names = numeric_features + ohe_cols

# fetch importances from regressor
reg = pipeline.named_steps['regressor']
if hasattr(reg, "feature_importances_"):
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(20)
    print("\nTop features:\n", top)
    # plot top 10
    top.head(10).sort_values().plot(kind='barh', figsize=(8,6))
    plt.title("Top 10 features")
    plt.show()
else:
    print("Regressor does not provide feature_importances_.")

# ---- Optional: Try XGBoost quickly if available ----
if xgb_available:
    print("Training XGBoost (quick)...")
    xgb_pipe = Pipeline([('preprocessor', preprocessor),
                         ('regressor', XGBRegressor(n_estimators=200, tree_method='hist', n_jobs=-1, random_state=42))])
    xgb_pipe.fit(X_train, y_train)
    y_pred_xgb = xgb_pipe.predict(X_test)
    if log_target:
        y_pred_xgb_inv = np.expm1(y_pred_xgb)
    else:
        y_pred_xgb_inv = y_pred_xgb
    print("XGB R2:", round(r2_score(y_test_inv, y_pred_xgb_inv), 4))
    print("XGB MAE:", round(mean_absolute_error(y_test_inv, y_pred_xgb_inv), 2))
    print("XGB RMSE:", round(np.sqrt(mean_squared_error(y_test_inv, y_pred_xgb_inv)), 2))
