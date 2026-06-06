"""Model inference and FastAPI endpoints for iterative AQI forecasting."""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.openmeteo_client import fetch_last_n_hours
from src.features.build_features import (
    ROLLING_MEAN_VARS,
    ROLLING_MEAN_WINDOWS,
    ROLLING_STD_VARS,
    ROLLING_STD_WINDOWS,
    build_features,
    get_feature_columns,
    load_training_feature_columns,
    training_feature_cols_path,
)
from src.utils.mongo_store import (
    DEFAULT_MODEL_NAME,
    get_latest_model_document,
    load_latest_model,
    read_features_since,
)

load_dotenv()
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")
TARGET_HORIZONS = [24, 48, 72]
FORECAST_STEPS = 96
HISTORY_DAYS = 30


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _normalize_artifact(obj) -> dict:
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj, "scaler": None, "features": None}


def load_model_local() -> dict:
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local model at {path}. Run training first.")
    return _normalize_artifact(joblib.load(path))


def load_model_mongodb(cfg: dict) -> dict:
    name = cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME)
    return _normalize_artifact(load_latest_model(name, cfg))


def resolve_feature_columns(cfg: dict, local: bool = False) -> list[str]:
    if os.path.isfile(training_feature_cols_path()):
        return load_training_feature_columns()
    if not local:
        try:
            name = cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME)
            cols = (get_latest_model_document(name, cfg).get("metadata") or {}).get("feature_cols")
            if cols:
                return cols
        except Exception as exc:
            log.warning("Could not load feature_cols from registry: %s", exc)
    return get_feature_columns()


def get_recent_features_mongodb(cfg: dict) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=HISTORY_DAYS)
    df = read_features_since(cutoff, cfg)
    if df.empty:
        raise RuntimeError("MongoDB feature collection is empty for the recent window.")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_recent_features_live(cfg: dict) -> pd.DataFrame:
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    return build_features(fetch_last_n_hours(lat, lon, n_hours=24 * 10)).sort_values("timestamp").reset_index(drop=True)


def get_recent_features_from_local_csv() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backfill.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local fallback file not found: {csv_path}")
    return pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _recompute_dynamic_features(sim: pd.DataFrame) -> pd.DataFrame:
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


def forecast_iterative(df: pd.DataFrame, model, scaler, features: list[str], steps: int = FORECAST_STEPS) -> list[dict]:
    sim = _recompute_dynamic_features(
        df.sort_values("timestamp").reset_index(drop=True).copy()
    )
    sim["timestamp"] = pd.to_datetime(sim["timestamp"])

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
        pred = max(0.0, min(float(model.predict(X)[0]), 500.0))

        next_time = sim["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
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
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def predict(local: bool = False) -> dict:
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
    if available.empty and local:
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
        "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "forecasts": forecasts,
        "hourly_forecast": [
            {"timestamp": p["timestamp"].isoformat(), "aqi_us": p["aqi_us"]} for p in hourly
        ],
        "latest_actual": latest_aqi,
        "latest_timestamp": str(latest_ts),
        "feature_row": latest_row[features].to_dict(orient="records")[0],
    }


@asynccontextmanager
async def _api_lifespan(_app: FastAPI):
    log.info("AQI Predictor API starting.")
    yield
    log.info("AQI Predictor API stopped.")


app = FastAPI(
    title="AQI Predictor API — Karachi",
    description="3-day AQI forecast via iterative one-hour-ahead prediction.",
    version="2.0.0",
    lifespan=_api_lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


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
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        import uvicorn
        uvicorn.run("src.serving.predict:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print(json.dumps(predict(local="--local" in sys.argv), indent=2, default=str))
