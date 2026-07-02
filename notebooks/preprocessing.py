import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, ENCODER_PATH,
    NUMERIC_FEATURES, CATEGORICAL_OHE_FEATURES,
    HIGH_CARDINALITY_FEATURES, BINARY_FEATURES, TARGET_COLUMN,
)
from utils.logger import get_logger
logger = get_logger(__name__)
HOLIDAY_MONTHS = {11, 12, 1}  # Nov, Dec, Jan

class DeliveryDateImputer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])
        df["_duration"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days
        self.carrier_median_days_ = (
            df.groupby("Carrier")["_duration"]
            .median()
            .fillna(df["_duration"].median())
        )
        self.global_median_ = df["_duration"].median()
        return self
      
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["Order_Date"] = pd.to_datetime(df["Order_Date"])
        df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"])
        mask = df["Delivery_Date"].isna()
        if mask.sum() > 0:
            logger.info(f"Imputing {mask.sum()} missing Delivery_Date values.")
            for idx in df[mask].index:
                carrier = df.at[idx, "Carrier"]
                days = self.carrier_median_days_.get(carrier, self.global_median_)
                df.at[idx, "Delivery_Date"] = df.at[idx, "Order_Date"] + pd.Timedelta(days=days)
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        # Dates
        order_dt = pd.to_datetime(df["Order_Date"])
        delivery_dt = pd.to_datetime(df["Delivery_Date"])
        # Customer return ratio (avoids division by zero)
        df["Customer_Return_Ratio"] = (
            df["Past_Returns"] / df["Past_Purchases"].clip(lower=1)
        ).round(4)
        # Delivery duration feature
        df["Expected_Delivery_Days"] = (delivery_dt - order_dt).dt.days.clip(lower=0)
        # Temporal features
        df["Order_Month"] = order_dt.dt.month
        df["Order_DayOfWeek"] = order_dt.dt.dayofweek
        df["Is_Weekend_Order"] = (df["Order_DayOfWeek"] >= 5).astype(int)
        df["Is_Holiday_Season"] = order_dt.dt.month.isin(HOLIDAY_MONTHS).astype(int)
        # Discount flag
        df["Has_Discount"] = (df["Discount_Percentage"] > 0).astype(int)
        return df
      
class IQROutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str], factor: float = 1.5):
        self.columns = columns
        self.factor = factor
    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {}
        for col in self.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            before = df[col].copy()
            df[col] = df[col].clip(lower, upper)
            capped = (before != df[col]).sum()
            if capped > 0:
                logger.info(f"IQR capping '{col}': {capped} values clipped.")
        return df
      
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns
    def fit(self, X: pd.DataFrame, y=None):
        self.freq_maps_ = {}
        n = len(X)
        for col in self.columns:
            self.freq_maps_[col] = X[col].value_counts(normalize=True).to_dict()
        return self
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for col in self.columns:
            df[f"{col}_Freq"] = df[col].map(self.freq_maps_[col]).fillna(0.0)
            df.drop(columns=[col], inplace=True)
        return df
# Column Transformer Sklearn-native for numeric + OHE
def build_column_transformer() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_OHE_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
# Full Preprocessing Flow              returns X, y as numpy arrays
def load_and_preprocess(
    path: Path = RAW_DATA_PATH,
    fit_encoders: bool = True,
    encoder_path: Path = ENCODER_PATH,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    #runnn preprocessing pipeline  Returns: X_transformed (np.ndarray), y (np.ndarray), column_transformer (fitted)
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Raw data shape: {df.shape}")
    imputer = DeliveryDateImputer()
    df = imputer.fit_transform(df)
    engineer = FeatureEngineer()
    df = engineer.fit_transform(df)
    capper = IQROutlierCapper(columns=["Price"])
    df = capper.fit_transform(df)
    freq_encoder = FrequencyEncoder(columns=HIGH_CARDINALITY_FEATURES)
    if fit_encoders:
        df = freq_encoder.fit_transform(df)
        joblib.dump({
            "delivery_imputer": imputer,
            "feature_engineer": engineer,
            "outlier_capper": capper,
            "freq_encoder": freq_encoder,
        }, encoder_path)
        logger.info(f"Encoders saved to {encoder_path}")
    else:
        saved = joblib.load(encoder_path)
        freq_encoder = saved["freq_encoder"]
        df = freq_encoder.transform(df)
    # Update BINARY_FEATURES list: add freq-encoded columns
    freq_cols = [f"{c}_Freq" for c in HIGH_CARDINALITY_FEATURES]
    #  Build column transformer  
    ct = build_column_transformer()
    # Add freq columns to binary (passthrough) section dynamically
    ct.transformers[-1] = ("bin", "passthrough", BINARY_FEATURES + freq_cols)
                  # Extract target
    y = df[TARGET_COLUMN].values
    if fit_encoders:
        X_transformed = ct.fit_transform(df)
        joblib.dump(ct, encoder_path.parent / "column_transformer.joblib")
        logger.info(f"Column transformer saved. X shape: {X_transformed.shape}")
    else:
        ct_saved = joblib.load(encoder_path.parent / "column_transformer.joblib")
        X_transformed = ct_saved.transform(df)
    logger.info(f"Return rate: {y.mean():.2%}")
    return X_transformed, y, ct

def preprocess_single(input_dict: dict, encoder_path: Path = ENCODER_PATH) -> np.ndarray:
    saved = joblib.load(encoder_path)
    ct = joblib.load(encoder_path.parent / "column_transformer.joblib")
    df = pd.DataFrame([input_dict])
    df = saved["delivery_imputer"].transform(df)
    df = saved["feature_engineer"].transform(df)
    df = saved["outlier_capper"].transform(df)
    df = saved["freq_encoder"].transform(df)
    freq_cols = [f"{c}_Freq" for c in HIGH_CARDINALITY_FEATURES]
    ct.transformers[-1] = ("bin", "passthrough", BINARY_FEATURES + freq_cols)
    return ct.transform(df)

if __name__ == "__main__":
    Path(ENCODER_PATH).parent.mkdir(parents=True, exist_ok=True)
    X, y, ct = load_and_preprocess()
    logger.info(f"Preprocessing complete. X: {X.shape}, y: {y.shape}")
