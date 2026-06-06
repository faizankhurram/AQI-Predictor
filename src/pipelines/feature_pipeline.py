"""Hourly feature ingest for GitHub Actions."""

import logging
import os
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.openmeteo_client import fetch_for_live_ingest
from src.features.build_features import build_features, drop_incomplete_features
from src.utils.mongo_store import delete_feature_rows_after, upsert_features

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

INSERT_LOOKBACK_HOURS = 48


def current_local_hour(timezone_name: str) -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo(timezone_name)).floor("h").tz_localize(None)


def load_config() -> dict:
    cfg_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run():
    cfg = load_config()
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    timezone_name = cfg["location"].get("timezone", "Asia/Karachi")

    try:
        raw = fetch_for_live_ingest(lat, lon, lookback_days=5)
    except Exception as exc:
        log.error("Open-Meteo fetch failed: %s", exc)
        sys.exit(1)

    clean = drop_incomplete_features(build_features(raw))
    if clean.empty:
        log.warning("No complete feature rows; skipping write.")
        return

    upper_bound = current_local_hour(timezone_name)
    cutoff = upper_bound - timedelta(hours=INSERT_LOOKBACK_HOURS)
    to_insert = clean[(clean["timestamp"] >= cutoff) & (clean["timestamp"] <= upper_bound)].copy()
    if to_insert.empty:
        log.warning("No rows in insert window; skipping write.")
        return

    deleted = delete_feature_rows_after(upper_bound, cfg)
    if deleted:
        log.info("Deleted %d future-dated rows after %s.", deleted, upper_bound)

    upserted = upsert_features(to_insert, cfg)
    log.info("Upsert submitted for %d rows.", upserted)


if __name__ == "__main__":
    run()
