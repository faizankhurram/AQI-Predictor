"""
Transforms raw merged DataFrame (air quality + weather) into model-ready features.
All operations are vectorised; no row-wise apply loops.
"""

import pandas as pd
import numpy as np


HORIZON_HOURS = [24, 48, 72]
LAG_HOURS = [1, 24]


def compute_aqi_us(pm2_5: pd.Series) -> pd.Series:
    """
    EPA breakpoint formula for US AQI from PM2.5 (µg/m³).
    Used as a fallback if the API already returns `aqi_us` directly — this
    column is kept for cross-validation against the API value.
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    aqi = pd.Series(np.nan, index=pm2_5.index)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = (pm2_5 >= c_lo) & (pm2_5 <= c_hi)
        aqi[mask] = ((i_hi - i_lo) / (c_hi - c_lo)) * (pm2_5[mask] - c_lo) + i_lo
    return aqi.round(0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with at minimum [timestamp, pm2_5, pm10, no2, o3,
                                        aqi_us (or computed), temperature_2m,
                                        relative_humidity_2m, wind_speed_10m]
    Output: Feature-rich DataFrame ready for Feature Store ingestion.
            Rows at the tail (within max horizon) have NaN targets — drop before training.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Use API-provided AQI if available, else compute from PM2.5
    if "aqi_us" not in df.columns or df["aqi_us"].isna().all():
        df["aqi_us"] = compute_aqi_us(df["pm2_5"])

    # Calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month

    # Lag features
    for lag in LAG_HOURS:
        df[f"aqi_lag_{lag}h"] = df["aqi_us"].shift(lag)

    # Change-rate features
    df["aqi_change_1h"] = df["aqi_us"].diff(1)
    df["aqi_change_24h"] = df["aqi_us"].diff(24)

    # Target labels (future AQI, shifted backwards = look-ahead)
    for h in HORIZON_HOURS:
        df[f"aqi_t_plus_{h}h"] = df["aqi_us"].shift(-h)

    # Rename no2/o3 if they come in as full names
    rename_map = {"nitrogen_dioxide": "no2", "ozone": "o3"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure date column exists for partitioning
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date.astype(str)

    return df


def get_feature_columns() -> list[str]:
    """Ordered list of input feature columns used by the model."""
    base = [
        "pm2_5", "pm10", "no2", "o3",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "hour", "day_of_week", "month",
        "aqi_lag_1h", "aqi_lag_24h",
        "aqi_change_1h", "aqi_change_24h",
    ]
    return base


def get_target_columns() -> list[str]:
    return [f"aqi_t_plus_{h}h" for h in HORIZON_HOURS]


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where any feature or target is NaN (lags + lead targets)."""
    cols = get_feature_columns() + get_target_columns()
    existing = [c for c in cols if c in df.columns]
    return df.dropna(subset=existing).reset_index(drop=True)
