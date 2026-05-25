"""
Inference module: loads the registered model + latest feature rows
and returns a 3-day (72 h) AQI forecast for Karachi.

Can run in two modes:
  - MongoDB mode (default): pulls model + features from MongoDB
  - Local mode (--local): uses models_artifacts/best_model.pkl + data/backfill.csv
"""

import os
import sys
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features.build_features import (
    load_training_feature_columns,
    get_feature_columns,
    build_features,
    training_feature_cols_path,
)
from src.data.openmeteo_client import fetch_last_n_hours
from src.utils.mongo_store import (
    DEFAULT_MODEL_NAME,
    load_latest_model,
    get_latest_model_document,
    read_features,
)

load_dotenv()
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")
TARGET_HORIZONS = [24, 48, 72]  # hours ahead


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_model_local():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No local model found at {path}. Run training_pipeline.py first.")
    return joblib.load(path)


def load_model_mongodb(cfg: dict):
    model_name = cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME)
    return load_latest_model(model_name, cfg)


def get_latest_features_mongodb(cfg: dict) -> pd.DataFrame:
    """Pull the most recent rows from MongoDB feature store."""
    df = read_features(cfg)
    if df.empty:
        raise RuntimeError("MongoDB feature collection is empty.")
    return df.sort_values("timestamp")


def get_latest_features_live(cfg: dict) -> pd.DataFrame:
    """Fetch the last 72 h from Open-Meteo and compute features (live mode)."""
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    raw = fetch_last_n_hours(lat, lon, n_hours=72)
    featured = build_features(raw)
    return featured.sort_values("timestamp")


def get_latest_features_from_local_csv() -> pd.DataFrame:
    """Fallback source when live API data is too short for lag features."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backfill.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local fallback file not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df.sort_values("timestamp")


def resolve_feature_columns(cfg: dict, local: bool = False) -> list[str]:
    """Use pruned columns from training (local JSON or MongoDB registry metadata)."""
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


def predict(local: bool = False) -> dict:
    """
    Returns a dict:
      {
        "generated_at": ISO timestamp,
        "forecasts": [
          {"horizon_h": 24, "aqi_us": float, "label": str},
          {"horizon_h": 48, "aqi_us": float, "label": str},
          {"horizon_h": 72, "aqi_us": float, "label": str},
        ],
        "latest_actual": float,
        "latest_timestamp": ISO str,
        "feature_row": dict,   # for SHAP
      }
    """
    cfg = load_config()

    if local:
        model = load_model_local()
        df = get_latest_features_live(cfg)
    else:
        model = load_model_mongodb(cfg)
        try:
            df = get_latest_features_mongodb(cfg)
        except Exception:
            log.warning("MongoDB feature read failed; falling back to live fetch.")
            df = get_latest_features_live(cfg)

    feature_cols = resolve_feature_columns(cfg, local=local)
    # Use the most recent complete row
    available = df.dropna(subset=feature_cols)
    if available.empty and local:
        # Live endpoint can return only ~24 future hours, which breaks 24h lag features.
        # In local mode we fallback to the latest complete row from backfill.csv.
        df = get_latest_features_from_local_csv()
        available = df.dropna(subset=feature_cols)
    if available.empty:
        raise RuntimeError("No complete feature rows available for inference.")

    latest_row = available.iloc[[-1]]
    X = latest_row[feature_cols].values

    preds = model.predict(X)[0]  # shape (3,)

    forecasts = []
    for i, h in enumerate(TARGET_HORIZONS):
        aqi_val = max(float(preds[i]), 0.0)
        forecasts.append({
            "horizon_h": h,
            "aqi_us": round(aqi_val, 1),
            "label": aqi_label(aqi_val),
        })

    latest_aqi = float(latest_row["aqi_us"].values[0]) if "aqi_us" in latest_row.columns else None

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "forecasts": forecasts,
        "latest_actual": latest_aqi,
        "latest_timestamp": str(latest_row["timestamp"].values[0]),
        "feature_row": latest_row[feature_cols].to_dict(orient="records")[0],
    }


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


if __name__ == "__main__":
    import json
    result = predict(local="--local" in sys.argv)
    print(json.dumps(result, indent=2, default=str))
