"""
Inference module: loads the registered {model, scaler, features} artifact and
produces a 3-day AQI forecast for Karachi via **iterative one-hour-ahead**
prediction (recursive multi-step forecasting).

Modes:
  - MongoDB mode (default): pulls model + recent features from MongoDB
  - Local mode (--local): uses models_artifacts/best_model.pkl + live/CSV features
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.build_features import (
    TARGET_COL,
    load_training_feature_columns,
    get_feature_columns,
    build_features,
    feature_columns_from_df,
    training_feature_cols_path,
    ROLLING_MEAN_WINDOWS,
    ROLLING_STD_WINDOWS,
    ROLLING_MEAN_VARS,
    ROLLING_STD_VARS,
)
from src.data.openmeteo_client import fetch_last_n_hours
from src.utils.mongo_store import (
    DEFAULT_MODEL_NAME,
    load_latest_model,
    get_latest_model_document,
    read_features_since,
)

load_dotenv()
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")
TARGET_HORIZONS = [24, 48, 72]   # hours ahead, derived from the iterative forecast
FORECAST_STEPS = 96              # iterative one-hour steps (4 days)
HISTORY_DAYS = 30                # context window for rolling features


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Model loading (artifact = {"model", "scaler", "features"}) ─────────────────
def _normalize_artifact(obj) -> dict:
    """Accept either the new dict payload or a bare estimator (legacy)."""
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "scaler": None, "features": None}


def load_model_local() -> dict:
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local model found at {path}. Run training_pipeline.py first.")
    return _normalize_artifact(joblib.load(path))


def load_model_mongodb(cfg: dict) -> dict:
    model_name = cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME)
    return _normalize_artifact(load_latest_model(model_name, cfg))


def resolve_feature_columns(cfg: dict, local: bool = False) -> list[str]:
    """Feature list from the training run (local JSON or MongoDB registry metadata)."""
    if os.path.isfile(training_feature_cols_path()):
        return load_training_feature_columns()
    if not local:
        try:
            model_name = cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME)
            doc = get_latest_model_document(model_name, cfg)
            cols = (doc.get("metadata") or {}).get("feature_cols")
            if cols:
                return cols
        except Exception as exc:
            log.warning("Could not load feature_cols from registry: %s", exc)
    return get_feature_columns()


# ── Feature sources ───────────────────────────────────────────────────────────
def get_recent_features_mongodb(cfg: dict) -> pd.DataFrame:
    """Pull the recent window of engineered rows from MongoDB (for rolling context)."""
    cutoff = datetime.utcnow() - timedelta(days=HISTORY_DAYS)
    df = read_features_since(cutoff, cfg)
    if df.empty:
        raise RuntimeError("MongoDB feature collection is empty for the recent window.")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_recent_features_live(cfg: dict) -> pd.DataFrame:
    """Fetch recent Open-Meteo data and engineer features (live/local mode)."""
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    raw = fetch_last_n_hours(lat, lon, n_hours=24 * 10)
    featured = build_features(raw)
    return featured.sort_values("timestamp").reset_index(drop=True)


def get_recent_features_from_local_csv() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backfill.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local fallback file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Iterative forecast ────────────────────────────────────────────────────────
def _recompute_dynamic_features(sim: pd.DataFrame) -> pd.DataFrame:
    """Recompute the time/rolling/derived features that change over the horizon."""
    hour = sim["timestamp"].dt.hour
    sim["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    sim["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(int)

    for window in ROLLING_MEAN_WINDOWS:
        for var in ROLLING_MEAN_VARS:
            col = f"{var}_rolling_mean_{window}h"
            if var in sim.columns and col in sim.columns:
                sim[col] = sim[var].shift(1).rolling(window=window, min_periods=1).mean()
    for window in ROLLING_STD_WINDOWS:
        for var in ROLLING_STD_VARS:
            col = f"{var}_rolling_std_{window}h"
            if var in sim.columns and col in sim.columns:
                sim[col] = sim[var].shift(1).rolling(window=window, min_periods=1).std()

    if "aqi_change_rate" in sim.columns and "aqi_us" in sim.columns:
        sim["aqi_change_rate"] = sim["aqi_us"].diff()
    if {"pm_ratio", "pm2_5", "pm10"}.issubset(sim.columns):
        sim["pm_ratio"] = np.where(sim["pm10"] > 0, sim["pm2_5"] / sim["pm10"], 0)
    return sim


def forecast_iterative(
    df: pd.DataFrame,
    model,
    scaler,
    features: list[str],
    steps: int = FORECAST_STEPS,
) -> list[dict]:
    """Recursive multi-step forecast: predict +1h, append, recompute, repeat."""
    sim = df.sort_values("timestamp").reset_index(drop=True).copy()
    sim["timestamp"] = pd.to_datetime(sim["timestamp"])
    sim = _recompute_dynamic_features(sim)

    predictions: list[dict] = []
    for _ in range(steps):
        feat_row = sim[features].iloc[-1].copy()
        for f in features:
            if pd.isna(feat_row[f]):
                valid = sim[f].dropna()
                feat_row[f] = valid.iloc[-1] if len(valid) else 0.0
        X = feat_row.values.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(pd.DataFrame([feat_row], columns=features))
        pred = float(model.predict(X)[0])
        pred = max(0.0, min(pred, 500.0))

        last_time = sim["timestamp"].iloc[-1]
        next_time = last_time + pd.Timedelta(hours=1)
        predictions.append({"timestamp": next_time, "aqi_us": round(pred, 1)})

        new_row = sim.iloc[-1].copy()
        new_row["timestamp"] = next_time
        new_row["aqi_us"] = pred
        sim = pd.concat([sim, pd.DataFrame([new_row])], ignore_index=True)
        sim = _recompute_dynamic_features(sim)

    return predictions


def aqi_label(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def predict(local: bool = False) -> dict:
    """
    Returns a dict (dashboard contract preserved):
      {
        "generated_at": ISO,
        "forecasts": [{"horizon_h": 24/48/72, "aqi_us": float, "label": str}],
        "hourly_forecast": [{"timestamp": ISO, "aqi_us": float}],   # full 96h
        "latest_actual": float,
        "latest_timestamp": ISO,
        "feature_row": dict,     # for SHAP
      }
    """
    cfg = load_config()

    if local:
        artifact = load_model_local()
        try:
            df = get_recent_features_live(cfg)
        except Exception:
            df = get_recent_features_from_local_csv()
    else:
        artifact = load_model_mongodb(cfg)
        try:
            df = get_recent_features_mongodb(cfg)
        except Exception:
            log.warning("MongoDB feature read failed; falling back to live fetch.")
            df = get_recent_features_live(cfg)

    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact.get("features") or resolve_feature_columns(cfg, local=local)
    features = [c for c in features if c in df.columns]
    if not features:
        raise RuntimeError("No model features present in the feature frame.")

    available = df.dropna(subset=features)
    if available.empty:
        if local:
            df = get_recent_features_from_local_csv()
            available = df.dropna(subset=[c for c in features if c in df.columns])
        if available.empty:
            raise RuntimeError("No complete feature rows available for inference.")

    seed = available.copy()
    latest_row = seed.iloc[[-1]]
    latest_aqi = float(latest_row["aqi_us"].values[0]) if "aqi_us" in latest_row.columns else None
    latest_ts = pd.to_datetime(latest_row["timestamp"].values[0])

    hourly = forecast_iterative(seed, model, scaler, features, steps=FORECAST_STEPS)

    forecasts = []
    for h in TARGET_HORIZONS:
        idx = min(h - 1, len(hourly) - 1)
        aqi_val = hourly[idx]["aqi_us"] if hourly else (latest_aqi or 0.0)
        forecasts.append({
            "horizon_h": h,
            "aqi_us": round(float(aqi_val), 1),
            "label": aqi_label(aqi_val),
        })

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "hourly_forecast": [
            {"timestamp": p["timestamp"].isoformat(), "aqi_us": p["aqi_us"]} for p in hourly
        ],
        "latest_actual": latest_aqi,
        "latest_timestamp": str(latest_ts),
        "feature_row": latest_row[features].to_dict(orient="records")[0],
    }


# ── FastAPI ───────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def _api_lifespan(_app: FastAPI):
    log.info("AQI Predictor API starting up.")
    yield
    log.info("AQI Predictor API shutting down.")


app = FastAPI(
    title="AQI Predictor API — Karachi",
    description="3-day Air Quality Index forecast via iterative one-hour-ahead prediction.",
    version="2.0.0",
    lifespan=_api_lifespan,
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
    try:
        return predict(local=False)
    except Exception as exc:
        log.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predict/local")
def predict_local_endpoint():
    try:
        return predict(local=True)
    except Exception as exc:
        log.error("Local prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import json
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run("src.serving.predict:app", host="0.0.0.0", port=8000, reload=True)
    else:
        result = predict(local="--local" in sys.argv)
        print(json.dumps(result, indent=2, default=str))
