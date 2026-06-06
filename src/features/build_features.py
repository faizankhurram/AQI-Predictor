"""Feature engineering: clean, IQR cap, engineer, prepare training frames."""

import json
import os

import numpy as np
import pandas as pd

TARGET_COL = "target"
NON_FEATURE_COLS = {"timestamp", "datetime", "date", TARGET_COL, "aqi_us"}

ROLLING_MEAN_WINDOWS = [36]
ROLLING_STD_WINDOWS = [72]
ROLLING_MEAN_VARS = ["pm2_5", "carbon_monoxide", "pm10"]
ROLLING_STD_VARS = ["pm2_5", "pm10"]
IQR_SKIP_COLS = {"wind_direction_10m", "cloud_cover", "relative_humidity_2m"}

CANONICAL_FEATURE_COLUMNS = [
    "pm2_5", "pm10", "carbon_monoxide", "carbon_dioxide", "no2",
    "sulphur_dioxide", "o3", "dust", "uv_index",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover",
    "hour_sin", "wind_u", "wind_v",
    "pm2_5_rolling_mean_36h", "carbon_monoxide_rolling_mean_36h", "pm10_rolling_mean_36h",
    "pm2_5_rolling_std_72h", "pm10_rolling_std_72h",
    "aqi_change_rate", "pm_ratio", "is_rush_hour", "precip_likelihood",
]

REQUIRED_RAW_INPUT_COLUMNS = ["timestamp", "pm2_5", "pm10", "aqi_us"]
OPTIONAL_RAW_INPUT_COLUMNS = [
    "carbon_monoxide", "carbon_dioxide", "no2", "sulphur_dioxide", "o3",
    "dust", "uv_index",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "cloud_cover",
]
RAW_INPUT_COLUMNS = REQUIRED_RAW_INPUT_COLUMNS + OPTIONAL_RAW_INPUT_COLUMNS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].ffill(limit=2)
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    df[num_cols] = df[num_cols].bfill().ffill()
    return df


def iqr_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in IQR_SKIP_COLS or col == "aqi_us":
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)

    if "wind_speed_10m" in df.columns and "wind_direction_10m" in df.columns:
        rad = np.deg2rad(pd.to_numeric(df["wind_direction_10m"], errors="coerce").fillna(0))
        df["wind_u"] = -df["wind_speed_10m"] * np.sin(rad)
        df["wind_v"] = -df["wind_speed_10m"] * np.cos(rad)
        df = df.drop(columns=["wind_direction_10m"])

    for window in ROLLING_MEAN_WINDOWS:
        for var in ROLLING_MEAN_VARS:
            if var in df.columns:
                df[f"{var}_rolling_mean_{window}h"] = (
                    df[var].shift(1).rolling(window=window, min_periods=1).mean()
                )
    for window in ROLLING_STD_WINDOWS:
        for var in ROLLING_STD_VARS:
            if var in df.columns:
                df[f"{var}_rolling_std_{window}h"] = (
                    df[var].shift(1).rolling(window=window, min_periods=1).std()
                )

    df = df.ffill().dropna()

    df["aqi_change_rate"] = df["aqi_us"].diff() if "aqi_us" in df.columns else np.nan
    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm_ratio"] = np.where(df["pm10"] > 0, df["pm2_5"] / df["pm10"], 0)
    else:
        df["pm_ratio"] = 0

    hour = df["timestamp"].dt.hour
    df["is_rush_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(int)

    if "relative_humidity_2m" in df.columns and "cloud_cover" in df.columns:
        df["precip_likelihood"] = df["relative_humidity_2m"] * df["cloud_cover"]
    else:
        df["precip_likelihood"] = 0

    df = df.bfill().ffill()
    if "date" not in df.columns:
        df["date"] = df["timestamp"].dt.date.astype(str)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(iqr_data(clean_data(df)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return prepare_data(df)


def feature_columns_from_df(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def get_feature_columns() -> list[str]:
    return list(CANONICAL_FEATURE_COLUMNS)


def get_target_columns() -> list[str]:
    return [TARGET_COL]


def drop_incomplete_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = feature_columns_from_df(df)
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df.reset_index(drop=True)
    return df.dropna(subset=existing).reset_index(drop=True)


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
    path = training_feature_cols_path(models_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return get_feature_columns()


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if not all(c in df.columns for c in REQUIRED_RAW_INPUT_COLUMNS):
        return drop_incomplete_features(df)

    has_engineered = all(c in df.columns for c in CANONICAL_FEATURE_COLUMNS)
    needs_rebuild = not has_engineered
    if not needs_rebuild:
        nan_rate = df[[c for c in CANONICAL_FEATURE_COLUMNS if c in df.columns]].isna().mean().max()
        needs_rebuild = bool(nan_rate > 0.05)

    if not needs_rebuild:
        return drop_incomplete_features(df)

    base_cols = [c for c in RAW_INPUT_COLUMNS if c in df.columns]
    featured = prepare_data(df[base_cols].copy())
    return drop_incomplete_features(featured)
