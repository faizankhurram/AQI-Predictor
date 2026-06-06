"""Open-Meteo air quality and weather client for Karachi."""

import time
from datetime import datetime, timedelta

import pandas as pd
import requests

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_VARS = [
    "pm2_5", "pm10", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "uv_index", "us_aqi",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "cloud_cover",
]
AQ_RENAME = {"nitrogen_dioxide": "no2", "ozone": "o3", "us_aqi": "aqi_us"}


def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise last_exc


def _parse_hourly(response: dict, rename: dict | None = None) -> pd.DataFrame:
    hourly = response.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    if rename:
        df.rename(columns=rename, inplace=True)
    return df


def fetch_air_quality(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(AQ_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    return _parse_hourly(_get_json(AQ_URL, params), AQ_RENAME)


def fetch_weather_forecast(lat: float, lon: float, days: int = 7) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "forecast_days": days,
        "timezone": "Asia/Karachi",
    }
    return _parse_hourly(_get_json(FORECAST_URL, params))


def fetch_weather_recent(lat: float, lon: float, past_days: int = 5, forecast_days: int = 1) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Karachi",
    }
    return _parse_hourly(_get_json(FORECAST_URL, params))


def fetch_weather_historical(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    return _parse_hourly(_get_json(ARCHIVE_URL, params))


def fetch_combined(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    is_historical: bool = True,
) -> pd.DataFrame:
    aq_df = fetch_air_quality(lat, lon, start_date, end_date)
    if is_historical:
        wx_df = fetch_weather_historical(lat, lon, start_date, end_date)
    else:
        days_ahead = max(
            (datetime.strptime(end_date, "%Y-%m-%d") - datetime.today()).days + 1,
            1,
        )
        wx_df = fetch_weather_forecast(lat, lon, days=days_ahead)
    merged = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    merged["date"] = merged["timestamp"].dt.date.astype(str)
    return merged


def fetch_for_live_ingest(lat: float, lon: float, lookback_days: int = 5) -> pd.DataFrame:
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)
    aq_df = fetch_air_quality(
        lat, lon, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
    )
    wx_df = fetch_weather_recent(lat, lon, past_days=lookback_days, forecast_days=1)
    merged = pd.merge(aq_df, wx_df, on="timestamp", how="inner")
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    merged["date"] = merged["timestamp"].dt.date.astype(str)
    return merged.reset_index(drop=True)


def fetch_last_n_hours(lat: float, lon: float, n_hours: int = 72) -> pd.DataFrame:
    df = fetch_for_live_ingest(lat, lon, lookback_days=max(3, (n_hours // 24) + 2))
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=n_hours)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)
