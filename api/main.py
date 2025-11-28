from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Pulsar Model Server")

MODEL_DIR = Path("models")
AVAILABLE_MODELS = {
    "lightgbm": "lightgbm.joblib",
    "xgboost": "xgboost.joblib",
    "svm": "svm.joblib",
    "random_forest": "random_forest.joblib",
    "logreg": "logreg.joblib",
}

class PulsarRequest(BaseModel):
    features: list[float]

class BatchRequest(BaseModel):
    rows: list[list[float]]

def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

    path = MODEL_DIR / AVAILABLE_MODELS[model_name]
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not trained/saved yet.")

    return joblib.load(path)

@app.get("/health")
def health():
    return {"status": "ok", "available_models": list(AVAILABLE_MODELS.keys())}

@app.post("/predict_proba")
def predict_proba(
        req: PulsarRequest,
        model_name: str = Query("lightgbm", description="Model selection")
    ):
    model = load_model(model_name)
    x = np.array([req.features], dtype=float)
    proba = float(model.predict_proba(x)[0,1])
    return {"pulsar_probability": proba}

@app.post("/predict_batch")
def predict_batch(
        req: BatchRequest,
        model_name: str = Query("lightgbm", description="Model selection")
    ):
    model = load_model(model_name)
    x = np.array(req.rows, dtype=float)

    if not np.isfinite(x).all():
        raise HTTPException(status_code=422, detail="Input contains NaN or infinite values.")

    probs = model.predict_proba(x)[:,1].tolist()
    return {"pulsar_probabilities": probs}
