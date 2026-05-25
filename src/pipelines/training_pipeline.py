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
from src.models.sklearn_trainer import train_and_evaluate, time_split, MODELS_DIR
from src.features.build_features import (
    get_target_columns,
    drop_incomplete_rows,
    prepare_training_frame,
    load_training_feature_columns,
    preprocess_training_splits,
    prune_correlated_features,
)
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
    avg = result["metrics"]["average"]
    model_dir = MODELS_DIR
    metrics_path = os.path.join(model_dir, "metrics.json")
    model_doc = save_model_artifact(
        name=cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME),
        model_path=result["model_path"],
        metrics_path=metrics_path,
        metadata={
            "best_name": result["best_name"],
            "rmse": avg["rmse"],
            "mae": avg["mae"],
            "r2": avg["r2"],
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

    # Train sklearn models
    corr_threshold = cfg.get("data", {}).get(
        "feature_correlation_threshold", 0.85
    )
    result = train_and_evaluate(
        df, test_days=test_days, correlation_threshold=corr_threshold
    )

    # Register to MongoDB (skip if using CSV-only local dev)
    if not csv_path:
        try:
            register_model_mongodb(cfg, result)
        except Exception as exc:
            log.warning("MongoDB model registration failed: %s — saved locally only.", exc)

    if with_tf:
        log.info("Training optional TensorFlow MLP...")
        try:
            from src.models.tf_trainer import train_mlp
            from src.models.sklearn_trainer import time_split
            import numpy as np

            target_cols = get_target_columns()
            train, test = time_split(df, test_days=test_days)
            train, test = preprocess_training_splits(
                train, test, load_training_feature_columns(), target_cols
            )
            feature_cols = load_training_feature_columns()
            X_train = train[feature_cols].values
            Y_train = train[target_cols].values
            X_test = test[feature_cols].values
            Y_test = test[target_cols].values

            _, tf_metrics, _, _ = train_mlp(X_train, Y_train, X_test, Y_test)
            sk_rmse = result["metrics"]["average"]["rmse"]
            tf_rmse = tf_metrics["average"]["rmse"]
            if tf_rmse < sk_rmse:
                log.info("TF-MLP (RMSE=%.2f) beats sklearn (RMSE=%.2f); consider registering TF model.", tf_rmse, sk_rmse)
            else:
                log.info("sklearn (RMSE=%.2f) still best — TF-MLP RMSE=%.2f; sklearn model retained.", sk_rmse, tf_rmse)
        except ImportError as e:
            log.warning("TensorFlow not available: %s", e)

    log.info("Training pipeline complete.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to local CSV backup")
    parser.add_argument("--with-tf", action="store_true", help="Also train TensorFlow MLP")
    parser.add_argument("--test-days", type=int, default=14, help="Holdout size in days")
    args = parser.parse_args()
    run(csv_path=args.csv, with_tf=args.with_tf, test_days=args.test_days)
