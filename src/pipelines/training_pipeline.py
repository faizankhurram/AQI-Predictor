"""Daily model training and MongoDB registration."""

import argparse
import logging
import os
import sys

import pandas as pd
import yaml
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.features.build_features import prepare_training_frame
from src.models.sklearn_trainer import MODELS_DIR, train_and_evaluate
from src.utils.mongo_store import DEFAULT_MODEL_NAME, read_features, save_model_artifact

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_from_mongodb(cfg: dict) -> pd.DataFrame:
    df = read_features(cfg)
    if df.empty:
        raise RuntimeError("MongoDB feature collection is empty. Run backfill first.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def register_model_mongodb(cfg: dict, result: dict):
    sel = result["metrics"]["selected"]
    model_doc = save_model_artifact(
        name=cfg.get("mongodb", {}).get("model_name", DEFAULT_MODEL_NAME),
        model_path=result["model_path"],
        metrics_path=os.path.join(MODELS_DIR, "metrics.json"),
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
    log.info("Model registered (id=%s).", model_doc["_id"])
    return model_doc


def run(csv_path: str | None = None):
    cfg = load_config()

    if csv_path:
        log.info("Loading CSV: %s", csv_path)
        df = prepare_training_frame(pd.read_csv(csv_path, parse_dates=["timestamp"]))
    else:
        log.info("Loading MongoDB feature store...")
        df = prepare_training_frame(load_from_mongodb(cfg))

    if df.empty:
        raise RuntimeError(
            "No training rows after feature preparation. "
            "Run: python run_pipeline.py backfill --days 365"
        )

    log.info("Training on %d rows (%s → %s)", len(df), df["timestamp"].min(), df["timestamp"].max())
    result = train_and_evaluate(df)

    if not csv_path:
        try:
            register_model_mongodb(cfg, result)
        except Exception as exc:
            log.warning("MongoDB registration failed: %s", exc)

    log.info("Training complete.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()
    run(csv_path=args.csv)
