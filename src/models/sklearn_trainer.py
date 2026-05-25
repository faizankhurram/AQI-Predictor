"""
Trains Linear Regression, Ridge, RandomForest, and XGBoost regressors on the
backfilled feature data.
Uses a MultiOutputRegressor wrapper so one model handles all three forecast
horizons (+24h, +48h, +72h) simultaneously.
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.features.build_features import (
    get_feature_columns,
    get_target_columns,
    preprocess_training_splits,
)
from src.models.metrics import evaluate_all_horizons, save_metrics

log = logging.getLogger(__name__)

HORIZON_LABELS = ["aqi_t_plus_24h", "aqi_t_plus_48h", "aqi_t_plus_72h"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")


def build_linear_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", MultiOutputRegressor(LinearRegression())),
    ])


def time_split(df: pd.DataFrame, test_days: int = 14):
    """Time-based train/test split — no shuffle, avoids leakage."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]
    return train, test


def build_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("model", MultiOutputRegressor(Ridge(alpha=2.0))),
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("model", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("model", MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                tree_method="hist",
            )
        )),
    ])


def train_and_evaluate(df: pd.DataFrame, test_days: int = 14) -> dict:
    """
    Trains candidate models, evaluates on time-split holdout, and saves the
    best model to disk.  Returns a dict with results + best model info.
    """
    feature_cols = get_feature_columns()
    target_cols = get_target_columns()

    train, test = time_split(df, test_days=test_days)
    log.info("Train: %d rows  |  Test: %d rows", len(train), len(test))

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Not enough data for a {test_days}-day holdout "
            f"(train={len(train)}, test={len(test)}). "
            "Run backfill to load more history or reduce --test-days."
        )

    train, test = preprocess_training_splits(train, test, feature_cols, target_cols)

    X_train = train[feature_cols].values
    Y_train = train[target_cols].values
    X_test = test[feature_cols].values
    Y_test = test[target_cols].values

    candidates = {
        "linear_regression": build_linear_pipeline(),
        "ridge": build_ridge_pipeline(),
        "random_forest": build_rf_pipeline(),
        "xgboost": build_xgb_pipeline(),
    }

    results = {}
    for name, pipe in candidates.items():
        log.info("Training %s...", name)
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)

        true_dict = {col: Y_test[:, i] for i, col in enumerate(target_cols)}
        pred_dict = {col: Y_pred[:, i] for i, col in enumerate(target_cols)}
        metrics = evaluate_all_horizons(true_dict, pred_dict)
        results[name] = {"pipeline": pipe, "metrics": metrics}
        log.info("%s avg RMSE=%.2f MAE=%.2f R²=%.3f",
                 name,
                 metrics["average"]["rmse"],
                 metrics["average"]["mae"],
                 metrics["average"]["r2"])

    best_name = min(results, key=lambda k: results[k]["metrics"]["average"]["rmse"])
    log.info("Best model: %s", best_name)

    os.makedirs(MODELS_DIR, exist_ok=True)
    best_pipe = results[best_name]["pipeline"]
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipe, model_path)
    log.info("Saved best model → %s", model_path)

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    save_metrics(
        {name: r["metrics"] for name, r in results.items()},
        metrics_path,
    )
    log.info("Saved metrics → %s", metrics_path)

    return {
        "best_name": best_name,
        "best_pipeline": best_pipe,
        "metrics": results[best_name]["metrics"],
        "all_metrics": {n: r["metrics"] for n, r in results.items()},
        "model_path": model_path,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }
