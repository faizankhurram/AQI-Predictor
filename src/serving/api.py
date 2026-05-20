"""
FastAPI serving layer.

Endpoints:
  GET  /health          — liveness check
  GET  /predict         — run inference, return 3-day forecast JSON
  GET  /predict/local   — same but uses local model + live Open-Meteo (no Hopsworks)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.serving.predict import predict

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AQI Predictor API starting up.")
    yield
    log.info("AQI Predictor API shutting down.")


app = FastAPI(
    title="AQI Predictor API — Karachi",
    description="3-day Air Quality Index forecast powered by Open-Meteo + Hopsworks + scikit-learn.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict_endpoint():
    """Fetch latest features from Hopsworks FG and return 3-day forecast."""
    try:
        result = predict(local=False)
        return result
    except Exception as exc:
        log.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/predict/local")
def predict_local():
    """Use local model artifact + live Open-Meteo data (no Hopsworks needed)."""
    try:
        result = predict(local=True)
        return result
    except Exception as exc:
        log.error("Local prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
