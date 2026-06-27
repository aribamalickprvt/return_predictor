"""
FastAPI Application
====================
Exposes a REST endpoint for e-commerce return prediction.

Endpoints:
  GET  /          — Health check
  GET  /health    — Detailed health check with model status
  POST /predict   — Single order return prediction
"""

import sys
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MODEL_ARTIFACT_PATH, API_HOST, API_PORT
from utils.logger import get_logger
from api.schemas import PredictionRequest, PredictionResponse
from api.inference import predict, load_model

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="E-Commerce Smart Return Predictor API",
    description=(
        "Predicts the probability that a customer will return a purchased item "
        "using a trained XGBoost classifier. Includes risk tiers and demarketing "
        "recommendations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pre-warm model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading ML model into memory...")
    load_model()
    logger.info("Model ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {"message": "Smart Return Predictor API is running.", "docs": "/docs"}


@app.get("/health", tags=["health"])
def health_check():
    model_exists = MODEL_ARTIFACT_PATH.exists()
    eval_path = MODEL_ARTIFACT_PATH.parent / "evaluation_results.json"

    eval_results = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

    return {
        "status": "healthy" if model_exists else "degraded",
        "model_loaded": model_exists,
        "model_path": str(MODEL_ARTIFACT_PATH),
        "evaluation_summary": eval_results,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_return(request: PredictionRequest):
    """
    Accepts a single order's features and returns:
    - **return_probability**: float between 0 and 1
    - **risk_tier**: LOW | MEDIUM | HIGH
    - **recommended_action**: suggested demarketing response
    """
    try:
        result = predict(request.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
