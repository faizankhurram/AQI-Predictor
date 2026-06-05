"""
Trains an ensemble of regressors to predict the **next hour's** US AQI, then
selects the best by R² (tie-break RMSE, MAE), retrains it on the full dataset,
and persists the model + scaler + feature list as one artifact.

Models: RandomForest, GradientBoosting, XGBoost, SVR, Ridge.
Scaler: MinMaxScaler. Split: time-ordered 80/20.

Multi-day forecasts are produced at serving time by iterative one-hour-ahead
prediction (see src/serving/predict.py).
"""

import json
import os
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from src.features.build_features import (
    TARGET_COL,
    feature_columns_from_df,
    save_training_feature_columns,
)

log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")


# ── Metrics ───────────────────────────────────────────────────────────────────
def calculate_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Model factory ─────────────────────────────────────────────────────────────
def _model_factory() -> dict:
    factory = {
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=15, max_features=0.7,
            min_samples_split=5, min_samples_leaf=4, random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.5, max_depth=3,
            min_samples_split=15, min_samples_leaf=5, subsample=0.7, random_state=42,
        ),
        "svr": lambda: SVR(kernel="rbf", C=10, gamma=0.001, epsilon=0.1),
        "ridge": lambda: Ridge(alpha=1.0),
    }
    if XGBRegressor is not None:
        factory["xgboost"] = lambda: XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.5, random_state=42, n_jobs=-1,
        )
    return factory


def time_split(df: pd.DataFrame, test_days: int = 14):
    """Time-ordered split (kept for callers); test_days unused with 80/20 default."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]


def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create next-hour AQI target and resolve resulting NaNs (reference logic)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[TARGET_COL] = df["aqi_us"].shift(-1)
    df = df.ffill().dropna().reset_index(drop=True)
    return df


def train_and_evaluate(df: pd.DataFrame, test_days: int = 14, **_ignored) -> dict:
    """
    Train candidates on a time-ordered 80/20 split, pick the best by R²,
    retrain it on the full dataset, and persist {model, scaler, features}.
    """
    df = _build_target(df)

    feature_cols = feature_columns_from_df(df)
    X = df[feature_cols]
    y = df[TARGET_COL]

    n = len(df)
    if n < 50:
        raise ValueError(f"Not enough rows to train ({n}). Run backfill first.")
    split_idx = int(n * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    factory = _model_factory()
    all_metrics: dict[str, dict] = {}
    for name, make in factory.items():
        try:
            model = make()
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            m = calculate_metrics(y_test, preds)
            all_metrics[name] = m
            log.info("%-18s RMSE=%.2f MAE=%.2f R2=%.3f", name, m["rmse"], m["mae"], m["r2"])
        except Exception as exc:
            log.warning("Model %s failed to train: %s", name, exc)

    if not all_metrics:
        raise RuntimeError("No models trained successfully.")

    # Best: highest R², tie-break lowest RMSE, then MAE.
    best_name = sorted(
        all_metrics,
        key=lambda k: (-all_metrics[k]["r2"], all_metrics[k]["rmse"], all_metrics[k]["mae"]),
    )[0]
    log.info("Best model: %s (R2=%.3f)", best_name, all_metrics[best_name]["r2"])

    # Retrain best on the full dataset with a fresh scaler.
    final_scaler = MinMaxScaler()
    X_full_scaled = final_scaler.fit_transform(X)
    best_model = factory[best_name]()
    best_model.fit(X_full_scaled, y)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    payload = {"model": best_model, "scaler": final_scaler, "features": feature_cols}
    joblib.dump(payload, model_path)
    log.info("Saved model artifact → %s", model_path)

    save_training_feature_columns(feature_cols, MODELS_DIR)

    selected = all_metrics[best_name]
    metrics_out = {
        "selected_model": best_name,
        "selected": selected,
        "average": selected,  # single-horizon; kept for downstream compatibility
        "all_models": all_metrics,
    }
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    save_metrics(metrics_out, metrics_path)
    log.info("Saved metrics → %s", metrics_path)

    return {
        "best_name": best_name,
        "model_path": model_path,
        "metrics": metrics_out,
        "all_metrics": all_metrics,
        "feature_cols": feature_cols,
        "target_cols": [TARGET_COL],
    }
