"""
Trains Linear Regression, Ridge, RandomForest, and XGBoost regressors on the
backfilled feature data.
Uses a MultiOutputRegressor wrapper so one model handles all three forecast
horizons (+24h, +48h, +72h) simultaneously.
"""

import json
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.features.build_features import (
    get_feature_columns,
    get_target_columns,
    preprocess_training_splits,
    prune_correlated_features,
    save_training_feature_columns,
    DEFAULT_CORRELATION_THRESHOLD,
)

log = logging.getLogger(__name__)

HORIZON_LABELS = ["aqi_t_plus_24h", "aqi_t_plus_48h", "aqi_t_plus_72h"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")


# ── Evaluation metrics (formerly metrics.py) ──────────────────────────────────

def evaluate(y_true, y_pred, horizon_label: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"horizon": horizon_label, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_all_horizons(y_true_dict: dict, y_pred_dict: dict) -> dict:
    results = {}
    rmses, maes, r2s = [], [], []
    for label in y_true_dict:
        m = evaluate(y_true_dict[label], y_pred_dict[label], label)
        results[label] = m
        rmses.append(m["rmse"])
        maes.append(m["mae"])
        r2s.append(m["r2"])
    results["average"] = {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
    }
    return results


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Model pipelines ───────────────────────────────────────────────────────────

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


def train_and_evaluate(
    df: pd.DataFrame,
    test_days: int = 14,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> dict:
    """
    Trains candidate models, evaluates on time-split holdout, and saves the
    best model to disk.  Returns a dict with results + best model info.
    """
    all_feature_cols = get_feature_columns()
    target_cols = get_target_columns()

    train, test = time_split(df, test_days=test_days)
    log.info("Train: %d rows  |  Test: %d rows", len(train), len(test))

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Not enough data for a {test_days}-day holdout "
            f"(train={len(train)}, test={len(test)}). "
            "Run backfill to load more history or reduce --test-days."
        )

    train, test = preprocess_training_splits(train, test, all_feature_cols, target_cols)

    feature_cols = prune_correlated_features(
        train,
        all_feature_cols,
        target_col="aqi_t_plus_24h",
        threshold=correlation_threshold,
    )
    dropped = sorted(set(all_feature_cols) - set(feature_cols))
    if dropped:
        log.info(
            "Dropped %d correlated features (|r|>=%.2f): %s",
            len(dropped),
            correlation_threshold,
            dropped,
        )
    log.info("Training with %d features", len(feature_cols))

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

    cols_path = save_training_feature_columns(feature_cols, MODELS_DIR)
    log.info("Saved feature column list → %s", cols_path)

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
