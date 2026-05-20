"""
Evaluation metrics for AQI regression models.
All functions accept arrays/Series and return plain Python floats.
"""

import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate(y_true, y_pred, horizon_label: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"horizon": horizon_label, "rmse": rmse, "mae": mae, "r2": r2}


def evaluate_all_horizons(y_true_dict: dict, y_pred_dict: dict) -> dict:
    """
    y_true_dict / y_pred_dict: {horizon_label: array_like}
    Returns per-horizon metrics + an 'average' summary.
    """
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
