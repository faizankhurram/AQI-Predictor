"""
Training pipeline — runs daily via GitHub Actions.

1. Reads data from Hopsworks Feature Group (or local CSV backup if --csv flag used).
2. Trains Ridge + RandomForest; evaluates on time-split holdout.
3. Registers the best model in the Hopsworks Model Registry.
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
from src.features.build_features import get_feature_columns, get_target_columns, drop_incomplete_rows

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_from_hopsworks(cfg: dict) -> pd.DataFrame:
    from src.utils.hopsworks_login import login_hopsworks

    project = login_hopsworks()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=cfg["hopsworks"]["feature_group_name"],
        version=cfg["hopsworks"]["feature_group_version"],
    )
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def register_model_hopsworks(cfg: dict, result: dict):
    """Push the best sklearn model artifact to Hopsworks Model Registry."""
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    from src.utils.hopsworks_login import login_hopsworks

    project = login_hopsworks()
    mr = project.get_model_registry()

    avg = result["metrics"]["average"]
    model_dir = MODELS_DIR

    hw_model = mr.sklearn.create_model(
        name=cfg["hopsworks"]["model_name"],
        metrics={
            "rmse": avg["rmse"],
            "mae": avg["mae"],
            "r2": avg["r2"],
        },
        description=f"Best model: {result['best_name']} | "
                    f"RMSE={avg['rmse']:.2f} MAE={avg['mae']:.2f} R²={avg['r2']:.3f}",
        input_example=None,
        model_schema=None,
    )
    hw_model.save(model_dir)
    log.info("Model registered in Hopsworks Model Registry.")
    return hw_model


def run(csv_path: str | None = None, with_tf: bool = False, test_days: int = 14):
    cfg = load_config()

    if csv_path:
        log.info("Loading data from local CSV: %s", csv_path)
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = drop_incomplete_rows(df)
    else:
        log.info("Loading data from Hopsworks Feature Group...")
        df = load_from_hopsworks(cfg)
        df = drop_incomplete_rows(df)

    log.info("Dataset: %d rows, %s → %s",
             len(df), df["timestamp"].min(), df["timestamp"].max())

    # Train sklearn models
    result = train_and_evaluate(df, test_days=test_days)

    # Register to Hopsworks (skip if using CSV-only local dev)
    if not csv_path:
        try:
            register_model_hopsworks(cfg, result)
        except Exception as exc:
            log.warning("Model Registry registration failed: %s — saved locally only.", exc)

    if with_tf:
        log.info("Training optional TensorFlow MLP...")
        try:
            from src.models.tf_trainer import train_mlp
            from src.models.sklearn_trainer import time_split
            import numpy as np

            feature_cols = get_feature_columns()
            target_cols = get_target_columns()
            train, test = time_split(df, test_days=test_days)
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
