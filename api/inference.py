"""
Inference Engine
================
Loads the trained XGBoost model and preprocessors, runs predictions.
"""

import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_ARTIFACT_PATH, ENCODER_PATH, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD
from utils.logger import get_logger
from notebooks.preprocessing import preprocess_single

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model():
    """Loads and caches the trained XGBoost model."""
    logger.info(f"Loading model from {MODEL_ARTIFACT_PATH}")
    model = joblib.load(MODEL_ARTIFACT_PATH)
    return model


def classify_risk(prob: float) -> tuple[str, str]:
    """Maps a probability to a risk tier and recommended action."""
    if prob < LOW_RISK_THRESHOLD:
        return "LOW", "Standard fulfillment — no intervention needed."
    elif prob < HIGH_RISK_THRESHOLD:
        return "MEDIUM", "Send a size guide / review prompt before shipping."
    else:
        return "HIGH", "Hold shipment for review. Consider size confirmation email or refund offer at checkout."


def predict(input_dict: dict) -> dict:
    """
    Preprocesses input and returns prediction dict.

    Args:
        input_dict: Raw order feature dict (matches PredictionRequest schema)

    Returns:
        dict with return_probability, risk_tier, recommended_action
    """
    model = load_model()
    X = preprocess_single(input_dict, ENCODER_PATH)
    prob = float(model.predict_proba(X)[0][1])
    risk, action = classify_risk(prob)

    logger.info(f"Prediction: prob={prob:.4f} | risk={risk}")
    return {
        "return_probability": round(prob, 4),
        "risk_tier": risk,
        "recommended_action": action,
        "model_version": "xgboost-v1",
    }
