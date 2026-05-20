"""
Live feature pipeline — runs hourly via GitHub Actions.
Fetches the last 72 h of Open-Meteo data, computes features,
and upserts new rows into the Hopsworks Feature Group.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.openmeteo_client import fetch_combined
from src.features.build_features import build_features, drop_incomplete_rows

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run():
    cfg = load_config()
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]
    fg_name = cfg["hopsworks"]["feature_group_name"]
    fg_version = cfg["hopsworks"]["feature_group_version"]
    project_name = os.environ["HOPSWORKS_PROJECT"]
    api_key = os.environ["HOPSWORKS_API_KEY"]

    # Fetch last 72 h (guarantees lag features are calculable for last 24 h)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=72)
    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    log.info("Fetching Open-Meteo data %s → %s", start_date, end_date)
    try:
        raw = fetch_combined(lat, lon, start_date, end_date, is_historical=False)
    except Exception as exc:
        log.error("Open-Meteo fetch failed: %s", exc)
        sys.exit(1)

    log.info("Raw rows fetched: %d", len(raw))
    featured = build_features(raw)
    clean = drop_incomplete_rows(featured)
    log.info("Rows after feature build + drop nulls: %d", len(clean))

    if clean.empty:
        log.warning("No complete rows to insert — skipping Feature Store write.")
        return

    from src.utils.hopsworks_login import login_hopsworks

    project = login_hopsworks(project=project_name, api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly AQI features for Karachi",
    )

    # Hopsworks deduplicates on primary key on insert
    log.info("Inserting %d rows into Feature Group '%s'", len(clean), fg_name)
    fg.insert(clean, write_options={"wait_for_job": True})
    log.info("Done.")


if __name__ == "__main__":
    run()
