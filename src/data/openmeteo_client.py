"""
Fetches hourly air-quality and weather data from Open-Meteo for Karachi.

Air-quality endpoint: https://air-quality-api.open-meteo.com/v1/air-quality
Weather forecast endpoint: https://api.open-meteo.com/v1/forecast
Historical weather endpoint: https://archive-api.open-meteo.com/v1/archive
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time


AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "us_aqi"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]


def _parse_hourly(response: dict, rename: dict | None = None) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON block to a tidy DataFrame."""
    hourly = response.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    if rename:
        df.rename(columns=rename, inplace=True)
    return df


def fetch_air_quality(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch hourly air-quality variables for a date range.
    Returns DataFrame with columns: timestamp, pm2_5, pm10, no2, o3, aqi_us
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AQ_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(AQ_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rename = {
        "nitrogen_dioxide": "no2",
        "ozone": "o3",
        "us_aqi": "aqi_us",
    }
    return _parse_hourly(data, rename)


def fetch_weather_forecast(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    """
    Fetch hourly weather forecast for the next `days` days.
    Used in the serving/inference step.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "forecast_days": days,
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_hourly(resp.json())


def fetch_weather_historical(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical weather from the ERA5 archive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_hourly(resp.json())


def fetch_combined(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    is_historical: bool = True,
) -> pd.DataFrame:
    """
    Merge air-quality and weather DataFrames on timestamp for a date range.
    For recent/live data use the forecast endpoint; for backfill use archive.
    """
    aq_df = fetch_air_quality(lat, lon, start_date, end_date)

    if is_historical:
        wx_df = fetch_weather_historical(lat, lon, start_date, end_date)
    else:
        days_ahead = (
            datetime.strptime(end_date, "%Y-%m-%d") - datetime.today()
        ).days + 1
        days_ahead = max(days_ahead, 1)
        wx_df = fetch_weather_forecast(lat, lon, days=days_ahead)

    merged = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    merged["date"] = merged["timestamp"].dt.date.astype(str)
    return merged


def fetch_for_live_ingest(lat: float, lon: float, lookback_days: int = 5) -> pd.DataFrame:
    """
    Fetch enough history for lag-24h features, then merge today's forecast slice.

    Air-quality API often returns only ~24h when using a short forecast window; we pull
    multi-day historical AQ + archive weather, then append today from forecast.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    yesterday = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")

    hist = fetch_combined(lat, lon, start_date, yesterday, is_historical=True)
    frames = [hist]

    try:
        today = fetch_combined(lat, lon, end_date, end_date, is_historical=False)
        if not today.empty:
            frames.append(today)
    except requests.RequestException:
        pass

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged["date"] = merged["timestamp"].dt.date.astype(str)
    return merged.reset_index(drop=True)


def fetch_last_n_hours(lat: float, lon: float, n_hours: int = 72) -> pd.DataFrame:
    """
    Convenience wrapper: fetch the last n_hours of combined data.
    Used by the live feature pipeline.
    """
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=n_hours)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")
    df = fetch_for_live_ingest(lat, lon, lookback_days=max(3, (n_hours // 24) + 2))
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=n_hours)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)
