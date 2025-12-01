from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pulsar Model Server")

# ---------------------------
# Model registry
# ---------------------------
MODEL_DIR = Path("models")
AVAILABLE_MODELS = {
    "lightgbm": "lightgbm.joblib",
    "xgboost": "xgboost.joblib",
    "svm": "svm.joblib",
    "random_forest": "random_forest.joblib",
    "logreg": "logreg.joblib",
}

# ---------------------------
# CORS – allows Streamlit to connect
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # loosen later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Request Models
# ---------------------------
class PulsarRequest(BaseModel):
    features: list[float]

class BatchRequest(BaseModel):
    rows: list[list[float]]

# ---------------------------
# Cached model loader for speed
# ---------------------------
@lru_cache(maxsize=10)
def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown model '{model_name}'. Available models: {list(AVAILABLE_MODELS.keys())}")

    model_path = MODEL_DIR / AVAILABLE_MODELS[model_name]

    if not model_path.exists():
        raise HTTPException(status_code=400,
                            detail=f"Model '{model_name}' not trained/saved yet.")

    try:
        return joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "available_models": list(AVAILABLE_MODELS.keys())}

@app.get("/models")
def list_models():
    """Return list of trained models + which files exist."""
    response = {}
    for name, filename in AVAILABLE_MODELS.items():
        response[name] = (MODEL_DIR / filename).exists()
    return response

@app.post("/predict_proba")
def predict_proba(
    req: PulsarRequest,
    model_name: str = Query("lightgbm", description="Model selection")
):
    model = load_model(model_name)
    x = np.array([req.features], dtype=float)

    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Input contains NaN or infinite values.")

    proba = float(model.predict_proba(x)[0, 1])
    return {"model_used": model_name, "pulsar_probability": proba}

@app.post("/predict_batch")
def predict_batch(
    req: BatchRequest,
    model_name: str = Query("lightgbm", description="Model selection")
):
    model = load_model(model_name)
    x = np.array(req.rows, dtype=float)

    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Input contains NaN or infinite values.")

    probs = model.predict_proba(x)[:, 1].tolist()
    return {
        "model_used": model_name,
        "pulsar_probabilities": probs,
        "num_predictions": len(probs)
    }
