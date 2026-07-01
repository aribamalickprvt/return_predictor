"""
Endpoints: 2
  GET  /health     Detailed health check with model memory status
  POST /predict    Single order return prediction
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

model_loaded_successfully = False
@app.on_event("startup")
async def startup_event():
    global model_loaded_successfully
    logger.info("Pre-loading ML model into memory...")
    try:
        load_model()
        model_loaded_successfully = True
        logger.info("Model ready.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}", exc_info=True)
        model_loaded_successfully = False

@app.get("/health", tags=["health"])
def health_check():
    model_exists = MODEL_ARTIFACT_PATH.exists()
    is_healthy = model_exists and model_loaded_successfully
    
    eval_path = MODEL_ARTIFACT_PATH.parent / "evaluation_results.json"
    eval_results = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)
    return {
        "status": "healthy" if is_healthy else "degraded",
        "model_loaded_in_memory": model_loaded_successfully,
        "model_file_exists": model_exists,
        "evaluation_summary": eval_results,
    }

@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_return(request: PredictionRequest):
    if not model_loaded_successfully:
        raise HTTPException(status_code=503, detail="Model is not loaded or system is degraded.")
    try:
        result = predict(request.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
