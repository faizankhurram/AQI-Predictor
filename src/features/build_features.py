"""
Transforms raw merged DataFrame (air quality + weather) into model-ready features.
All operations are vectorised; no row-wise apply loops.
"""

import json
import os

import pandas as pd
import numpy as np


HORIZON_HOURS = [24, 48, 72]
LAG_HOURS = [1, 24]
ROLLING_WINDOW_24H = 24
POLLUTANT_ROLL_COLS = ["pm2_5", "pm10", "no2", "o3"]
DEFAULT_CORRELATION_THRESHOLD = 0.85

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
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- Weather interaction: smog index captures Karachi winter particulate trapping ---
    wind = df.get("wind_speed_10m", pd.Series(0.0, index=df.index)).fillna(0)
    humidity = df.get("relative_humidity_2m", pd.Series(0.0, index=df.index))
    temp = df.get("temperature_2m", pd.Series(0.0, index=df.index))
    df["smog_index"] = (humidity / (wind + 1)) * df["is_winter"]
    # Dispersion: dry + windy conditions reduce trapped pollution
    df["dispersion_index"] = wind * (100 - humidity.clip(0, 100)) / 100
    # Heat–humidity stress (stagnant hot humid periods in Karachi summers)
    df["heat_index"] = temp * humidity / 100

    # Cyclic day-of-week (replaces linear day_of_week for models)
    dow = df["timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Log-scaled coarse particulates (stabilizes spikes)
    if "pm10" in df.columns:
        df["log_pm10"] = np.log1p(df["pm10"].clip(lower=0))

    # Wind direction → U/V components (meteorological: direction wind comes from)
    if "wind_direction_10m" in df.columns:
        wd_rad = np.deg2rad(pd.to_numeric(df["wind_direction_10m"], errors="coerce").fillna(0))
        df["wind_u"] = -wind * np.sin(wd_rad)
        df["wind_v"] = -wind * np.cos(wd_rad)
    else:
        df["wind_u"] = 0.0
        df["wind_v"] = 0.0

    # 24-hour rolling means per pollutant (strong persistence signal for AQI)
    for col in POLLUTANT_ROLL_COLS:
        if col in df.columns:
            df[f"{col}_roll_24h"] = (
                df[col].rolling(window=ROLLING_WINDOW_24H, min_periods=ROLLING_WINDOW_24H).mean()
            )

    # --- Lag features ---
    df["aqi_lag_1h"] = df["aqi_us"].shift(1)
    df["aqi_lag_24h"] = df["aqi_us"].shift(24)

    # --- Rolling averages ---
    df["aqi_rolling_6h"] = df["aqi_us"].rolling(window=6, min_periods=1).mean()
    df["aqi_rolling_24h"] = df["aqi_us"].rolling(window=24, min_periods=1).mean()

    # --- Change-rate features (clipped to limit outlier leverage) ---
    df["aqi_change_1h"] = df["aqi_us"].diff(1).clip(-80, 80)
    df["aqi_change_24h"] = df["aqi_us"].diff(24).clip(-120, 120)
    lag_2h = df["aqi_us"].shift(2)
    df["aqi_change_rate"] = ((df["aqi_lag_1h"] - lag_2h) / (lag_2h + 0.1)).clip(-1.5, 1.5)

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
    """
    Ordered list of input feature columns used by the model.

    pm2_5 is intentionally excluded — it is almost collinear with aqi_us and lag
    features, which hurts linear models and inflates variance on tree models.
    """
    pollutant_rolls = [f"{c}_roll_24h" for c in POLLUTANT_ROLL_COLS]
    base = [
        "pm10", "log_pm10", "no2", "o3",
        *pollutant_rolls,
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_u", "wind_v",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dow_sin", "dow_cos", "is_winter",
        "smog_index", "dispersion_index", "heat_index",
        "aqi_lag_1h", "aqi_lag_24h", "aqi_rolling_6h", "aqi_rolling_24h",
        "aqi_change_1h", "aqi_change_24h", "aqi_change_rate",
    ]
    return base


# Columns winsorized from train quantiles before model fit
WINSORIZE_FEATURE_COLS = ["aqi_change_1h", "aqi_change_24h", "aqi_change_rate", "no2", "o3", "pm10"]
WINSOR_QUANTILES = (0.02, 0.98)
AQI_MIN, AQI_MAX = 0.0, 500.0


def preprocess_training_splits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Training-only cleaning: winsorize volatile columns, fill residual NaNs,
    and clip AQI targets to the valid EPA range.
    """
    train = train.copy()
    test = test.copy()

    for col in WINSORIZE_FEATURE_COLS:
        if col not in train.columns:
            continue
        lo, hi = train[col].quantile(WINSOR_QUANTILES[0]), train[col].quantile(WINSOR_QUANTILES[1])
        train[col] = train[col].clip(lo, hi)
        test[col] = test[col].clip(lo, hi)

    medians = train[feature_cols].median()
    train[feature_cols] = train[feature_cols].fillna(medians).replace([np.inf, -np.inf], 0.0)
    test[feature_cols] = test[feature_cols].fillna(medians).replace([np.inf, -np.inf], 0.0)

    for col in target_cols:
        train[col] = train[col].clip(AQI_MIN, AQI_MAX)
        test[col] = test[col].clip(AQI_MIN, AQI_MAX)

    return train, test


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
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Training frame is missing required columns: {missing}. "
            "Run prepare_training_frame() or backfill.py after a feature-schema change."
        )
    return df.dropna(subset=existing).reset_index(drop=True)


REQUIRED_RAW_INPUT_COLUMNS = [
    "timestamp",
    "pm2_5",
    "pm10",
    "no2",
    "o3",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]
OPTIONAL_RAW_INPUT_COLUMNS = ["wind_direction_10m"]
RAW_INPUT_COLUMNS = REQUIRED_RAW_INPUT_COLUMNS + OPTIONAL_RAW_INPUT_COLUMNS


def prune_correlated_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "aqi_t_plus_24h",
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> list[str]:
    """
    Drop one feature from each pair with |r| >= threshold, keeping the feature
    with stronger absolute correlation to the target (24h horizon proxy).
    """
    kept = [c for c in feature_cols if c in train.columns and train[c].notna().any()]
    if len(kept) < 2 or target_col not in train.columns:
        return kept

    while len(kept) >= 2:
        corr = train[kept + [target_col]].corr().abs()
        target_corr = corr[target_col]

        worst_pair = None
        worst_r = threshold
        for i, a in enumerate(kept):
            for b in kept[i + 1 :]:
                r = corr.loc[a, b]
                if r >= threshold and r >= worst_r:
                    worst_r = r
                    worst_pair = (a, b)

        if worst_pair is None:
            break

        a, b = worst_pair
        if target_corr[a] >= target_corr[b]:
            kept.remove(b)
        else:
            kept.remove(a)

    return kept


def training_feature_cols_path(models_dir: str | None = None) -> str:
    root = models_dir or os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts")
    return os.path.join(root, "feature_cols.json")


def save_training_feature_columns(cols: list[str], models_dir: str | None = None) -> str:
    path = training_feature_cols_path(models_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2)
    return path


def load_training_feature_columns(models_dir: str | None = None) -> list[str]:
    """Pruned columns from last training run; falls back to full feature list."""
    path = training_feature_cols_path(models_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return get_feature_columns()


def _pm25_needs_uncalibration(pm2_5: pd.Series, aqi_us: pd.Series) -> bool:
    """
    Return True when stored pm2_5 already includes the 1.42 calibration factor
    (so build_features should not multiply again).
    """
    sample = pd.DataFrame({"pm2_5": pm2_5, "aqi_us": aqi_us}).dropna()
    if len(sample) < 20:
        return False
    pm = sample["pm2_5"]
    aqi = sample["aqi_us"]
    err_if_raw = (compute_aqi_us(pm) - aqi).abs().median()
    err_if_cal = (compute_aqi_us(pm * PM25_CALIBRATION_FACTOR) - aqi).abs().median()
    return err_if_cal < err_if_raw


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a training-ready frame from MongoDB/CSV rows.

    Recomputes engineered features from raw AQ + weather columns when the stored
    schema is stale (e.g. after adding cyclic time features). This avoids 0-row
    training sets when old MongoDB documents lack new feature columns.
    """
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if not all(c in df.columns for c in REQUIRED_RAW_INPUT_COLUMNS):
        return drop_incomplete_rows(df)

    feat_cols = get_feature_columns()
    needs_rebuild = any(c not in df.columns for c in feat_cols)
    if not needs_rebuild:
        nan_rate = df[feat_cols].isna().mean().max()
        needs_rebuild = nan_rate > 0.05

    if not needs_rebuild:
        return drop_incomplete_rows(df)

    base_cols = [c for c in RAW_INPUT_COLUMNS if c in df.columns]
    base = df[base_cols].copy()
    if "aqi_us" in df.columns and _pm25_needs_uncalibration(df["pm2_5"], df["aqi_us"]):
        base["pm2_5"] = base["pm2_5"] / PM25_CALIBRATION_FACTOR

    featured = build_features(base)
    return drop_incomplete_rows(featured)
