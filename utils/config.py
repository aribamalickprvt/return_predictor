"""
Configuration & constants for the Return Predictor project.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "ecommerce_returns.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "xgb_pipeline.joblib"
ENCODER_PATH = MODELS_DIR / "target_encoders.joblib"

# ── Feature groups ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Price",
    "Discount_Percentage",
    "Account_Age_Days",
    "Customer_Return_Ratio",
    "Expected_Delivery_Days",
    "Order_Month",
    "Order_DayOfWeek",
    "Review_Score",
]

CATEGORICAL_OHE_FEATURES = [
    "Shipping_Method",
    "Carrier",
    "Category",
]

HIGH_CARDINALITY_FEATURES = [
    "ZIP_Code",
    "Product_ID",
]

BINARY_FEATURES = [
    "Bought_Two_Sizes",
    "Is_Weekend_Order",
    "Is_Holiday_Season",
    "Has_Discount",
]

TARGET_COLUMN = "Is_Returned"

# ── Model hyperparameters ──────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,   # handles class imbalance
    "eval_metric": "logloss",
    "random_state": 42,
    "use_label_encoder": False,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Streamlit ──────────────────────────────────────────────────────────────────
DASHBOARD_TITLE = "🛍️ Smart Return Predictor"
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60
