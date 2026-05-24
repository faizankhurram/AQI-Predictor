"""
Transforms raw merged DataFrame (air quality + weather) into model-ready features.
All operations are vectorised; no row-wise apply loops.
"""

import pandas as pd
import numpy as np


HORIZON_HOURS = [24, 48, 72]
LAG_HOURS = [1, 24]

# Sensor/satellite correction factor validated for South Asian low-cost sensors
# (consistent with peer-reviewed Karachi air quality studies).
PM25_CALIBRATION_FACTOR = 1.42


def compute_aqi_us(pm2_5: pd.Series) -> pd.Series:
    """
    2024 US EPA breakpoint formula for AQI from PM2.5 (µg/m³).
    Breakpoints updated per EPA's 2024 NAAQS revision (first band 0–9.0 µg/m³).
    Input pm2_5 should already be calibrated before calling this function.
    """
    # Truncate to one decimal place per EPA convention
    cp = np.floor(pd.to_numeric(pm2_5, errors="coerce") * 10) / 10
    aqi = pd.Series(np.nan, index=pm2_5.index)

    breakpoints = [
        (0.0,   9.0,   0,  50),
        (9.1,  35.4,  51, 100),
        (35.5,  55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        mask = (cp >= c_lo) & (cp <= c_hi)
        aqi[mask] = ((i_hi - i_lo) / (c_hi - c_lo)) * (cp[mask] - c_lo) + i_lo

    # Anything beyond 500.4 is capped at 500
    aqi[cp > 500.4] = 500.0
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

    # Apply PM2.5 calibration factor (corrects for sensor/satellite bias in South Asia)
    if "pm2_5" in df.columns:
        df["pm2_5"] = df["pm2_5"] * PM25_CALIBRATION_FACTOR

    # Recompute AQI from calibrated PM2.5 (overrides raw API value for consistency)
    df["aqi_us"] = compute_aqi_us(df["pm2_5"])

    # --- Time features (cyclic encoding avoids hour 23 → 0 discontinuity) ---
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    month = df["timestamp"].dt.month
    df["is_winter"] = month.isin([11, 12, 1, 2]).astype(int)

    # --- Weather interaction: smog index captures Karachi winter particulate trapping ---
    wind = df.get("wind_speed_10m", pd.Series(0.0, index=df.index))
    humidity = df.get("relative_humidity_2m", pd.Series(0.0, index=df.index))
    df["smog_index"] = (humidity / (wind + 1)) * df["is_winter"]

    # --- Lag features ---
    df["aqi_lag_1h"] = df["aqi_us"].shift(1)
    df["aqi_lag_24h"] = df["aqi_us"].shift(24)

    # --- Rolling average ---
    df["aqi_rolling_6h"] = df["aqi_us"].rolling(window=6, min_periods=1).mean()

    # --- Change-rate features ---
    df["aqi_change_1h"] = df["aqi_us"].diff(1)
    df["aqi_change_24h"] = df["aqi_us"].diff(24)
    lag_2h = df["aqi_us"].shift(2)
    df["aqi_change_rate"] = (df["aqi_lag_1h"] - lag_2h) / (lag_2h + 0.1)

    # Target labels (future AQI, shifted backwards = look-ahead)
    for h in HORIZON_HOURS:
        df[f"aqi_t_plus_{h}h"] = df["aqi_us"].shift(-h)

    # Rename no2/o3 if they come in as full names
    rename_map = {"nitrogen_dioxide": "no2", "ozone": "o3"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Guard against inf values produced by change-rate division
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    # Ensure date column exists for partitioning
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date.astype(str)

    return df


def get_feature_columns() -> list[str]:
    """Ordered list of input feature columns used by the model."""
    base = [
        "pm2_5", "pm10", "no2", "o3",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "hour_sin", "hour_cos", "day_of_week", "is_winter",
        "smog_index",
        "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_6h",
        "aqi_change_1h", "aqi_change_24h", "aqi_change_rate",
    ]
    return base


def get_target_columns() -> list[str]:
    return [f"aqi_t_plus_{h}h" for h in HORIZON_HOURS]


def drop_incomplete_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where input features are NaN (for hourly MongoDB ingest; targets may be NaN)."""
    cols = get_feature_columns()
    existing = [c for c in cols if c in df.columns]
    return df.dropna(subset=existing).reset_index(drop=True)


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where any feature or target is NaN (lags + lead targets). Used for training."""
    cols = get_feature_columns() + get_target_columns()
    existing = [c for c in cols if c in df.columns]
    return df.dropna(subset=existing).reset_index(drop=True)
