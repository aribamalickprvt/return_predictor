import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import (
    MODEL_ARTIFACT_PATH, ENCODER_PATH, XGB_PARAMS, RF_PARAMS,
)
from utils.logger import get_logger
from notebooks.preprocessing import load_and_preprocess
logger = get_logger(__name__)

def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logger.info(f"Before SMOTE: class distribution {np.bincount(y_train)}")
    sm = SMOTE(sampling_strategy=0.5, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info(f"After  SMOTE: class distribution {np.bincount(y_res)}")
    return X_res, y_res
def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    f1      = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    pr_auc  = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    results = {
        "model": name,
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm,
    }
    logger.info(
        f"\n{'='*50}\n{name} Results\n"
        f"  F1:        {f1:.4f}\n"
        f"  Precision: {prec:.4f}\n"
        f"  Recall:    {rec:.4f}\n"
        f"  PR-AUC:    {pr_auc:.4f}\n"
        f"  Confusion Matrix:\n{np.array(cm)}\n"
    )
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return results

def get_feature_importance(model: XGBClassifier, ct) -> pd.DataFrame:
    try:
        num_names = ct.transformers_[0][2]  # numeric features
        ohe = ct.transformers_[1][1]["ohe"]
        cat_names = list(ohe.get_feature_names_out(ct.transformers_[1][2]))
        bin_names = list(ct.transformers_[2][2])
        all_names = list(num_names) + cat_names + bin_names
        importance = model.feature_importances_
        if len(all_names) != len(importance):
            all_names = [f"feature_{i}" for i in range(len(importance))]
        df = pd.DataFrame({"feature": all_names, "importance": importance})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Could not extract named feature importances: {e}")
        return pd.DataFrame()

def train():
    Path(MODEL_ARTIFACT_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Load preprocessed data
    logger.info("Starting preprocessing...")
    X, y, ct = load_and_preprocess()
    # Train/test split 20/80
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    # Apply SMOTE
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    all_results = []
    #   Logistic Regression  
    logger.info("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_bal, y_train_bal)
    lr_results = evaluate_model("Logistic Regression", lr, X_test, y_test)
    all_results.append(lr_results)
    #  Random Forest  
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train_bal, y_train_bal)
    rf_results = evaluate_model("Random Forest", rf, X_test, y_test)
    all_results.append(rf_results)
    joblib.dump(rf, MODEL_ARTIFACT_PATH.parent / "rf_model.joblib")
    #  XGBoost (primary) 
    logger.info("Training XGBoost (primary model)...")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_results = evaluate_model("XGBoost", xgb, X_test, y_test)
    all_results.append(xgb_results)
    #      Save primary model
    joblib.dump(xgb, MODEL_ARTIFACT_PATH)
    logger.info(f"XGBoost model saved to {MODEL_ARTIFACT_PATH}")
    #  Feature importance 
    fi_df = get_feature_importance(xgb, ct)
    if not fi_df.empty:
        fi_path = MODEL_ARTIFACT_PATH.parent / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        logger.info(f"Feature importance saved to {fi_path}")
        logger.info(f"Top 10 features:\n{fi_df.head(10).to_string(index=False)}")
#Save evaluation results  
    results_path = MODEL_ARTIFACT_PATH.parent / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Evaluation results saved to {results_path}")
    return xgb, all_results

if __name__ == "__main__":
    train()
