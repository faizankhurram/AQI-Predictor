"""
Training pipeline — runs daily via GitHub Actions.

1. Reads data from MongoDB feature store (or local CSV backup if --csv flag used).
2. Trains Ridge + RandomForest; evaluates on time-split holdout.
3. Registers the best model artifact in MongoDB GridFS.
4. Optionally trains a TensorFlow MLP and registers it if it beats sklearn.

Usage:
    python src/pipelines/training_pipeline.py
    python src/pipelines/training_pipeline.py --csv data/backfill.csv   # local fallback
    python src/pipelines/training_pipeline.py --with-tf                 # also train TF model
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import logging

import pandas as pd
from dotenv import load_dotenv
import yaml
from src.models.sklearn_trainer import train_and_evaluate, MODELS_DIR
from src.features.build_features import prepare_training_frame
from src.utils.mongo_store import DEFAULT_MODEL_NAME, read_features, save_model_artifact

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_from_mongodb(cfg: dict) -> pd.DataFrame:
    df = read_features(cfg)
    if df.empty:
        raise RuntimeError("MongoDB feature collection is empty. Run backfill.py first.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def register_model_mongodb(cfg: dict, result: dict):
    """Push the best sklearn model artifact to MongoDB GridFS + registry metadata."""
    sel = result["metrics"]["selected"]
    model_dir = MODELS_DIR
    metrics_path = os.path.join(model_dir, "metrics.json")
    model_doc = save_model_artifact(
        name=cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME),
        model_path=result["model_path"],
        metrics_path=metrics_path,
        metadata={
            "best_name": result["best_name"],
            "rmse": sel["rmse"],
            "mae": sel["mae"],
            "r2": sel["r2"],
            "feature_cols": result["feature_cols"],
            "target_cols": result["target_cols"],
        },
        cfg=cfg,
    )
    log.info("Model registered in MongoDB model registry (id=%s).", model_doc["_id"])
    return model_doc


def run(csv_path: str | None = None, with_tf: bool = False, test_days: int = 14):
    cfg = load_config()

    if csv_path:
        log.info("Loading data from local CSV: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = prepare_training_frame(df)
    else:
        log.info("Loading data from MongoDB feature store...")
        df = load_from_mongodb(cfg)
        log.info("MongoDB documents loaded: %d", len(df))
        df = prepare_training_frame(df)

    if df.empty:
        raise RuntimeError(
            "No complete training rows after feature preparation. "
            "Run backfill (python src/pipelines/backfill.py --days 90) or ensure "
            "the feature collection has timestamp, pm2_5, weather columns, and enough history "
            "for 72h targets."
        )

    log.info("Dataset: %d rows, %s → %s",
             len(df), df["timestamp"].min(), df["timestamp"].max())

    # Train sklearn ensemble (next-hour AQI target).
    result = train_and_evaluate(df, test_days=test_days)

    # Register to MongoDB (skip if using CSV-only local dev)
    if not csv_path:
        try:
            register_model_mongodb(cfg, result)
        except Exception as exc:
            log.warning("MongoDB model registration failed: %s — saved locally only.", exc)

    if with_tf:
        log.info("--with-tf is no longer supported under the iterative-forecast pipeline; skipping.")

    log.info("Training pipeline complete.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to local CSV backup")
    parser.add_argument("--with-tf", action="store_true", help="Also train TensorFlow MLP")
    parser.add_argument("--test-days", type=int, default=14, help="Holdout size in days")
    args = parser.parse_args()
    run(csv_path=args.csv, with_tf=args.with_tf, test_days=args.test_days)
