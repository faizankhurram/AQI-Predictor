"""
Live feature pipeline — runs hourly via GitHub Actions.
Fetches the last 72 h of Open-Meteo data, computes features,
and upserts new rows into the Hopsworks Feature Group.
"""

import os
import sys

# Must run before `from src.*` (script mode does not add repo root to sys.path).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import logging
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
import yaml

try:
    from src.data.openmeteo_client import fetch_for_live_ingest
    from src.features.build_features import build_features, drop_incomplete_features
except ModuleNotFoundError as exc:
    _data_dir = os.path.join(_REPO_ROOT, "src", "data")
    raise SystemExit(
        f"Import failed ({exc}). repo_root={_REPO_ROOT!r}, "
        f"src/data exists={os.path.isdir(_data_dir)}. "
        "On GitHub Actions: use 'Run workflow' on branch main (not 'Re-run failed jobs' on an old run)."
    ) from exc

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Only upsert recent hours (dedupe on timestamp still applies for overlaps).
INSERT_LOOKBACK_HOURS = 48


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

    log.info("Fetching multi-day Open-Meteo window for lag features (lookback=5 days)")
    try:
        raw = fetch_for_live_ingest(lat, lon, lookback_days=5)
    except Exception as exc:
        log.error("Open-Meteo fetch failed: %s", exc)
        sys.exit(1)

    log.info("Raw rows fetched: %d (%s → %s)", len(raw), raw["timestamp"].min(), raw["timestamp"].max())
    featured = build_features(raw)
    # Ingest: require features only (targets need future hours; 24-row forecast window drops all rows).
    clean = drop_incomplete_features(featured)
    log.info("Rows with complete features: %d", len(clean))

    if clean.empty:
        log.warning("No rows with complete features — skipping Feature Store write.")
        return

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - timedelta(hours=INSERT_LOOKBACK_HOURS)
    to_insert = clean[clean["timestamp"] >= cutoff].copy()
    log.info("Rows to insert (last %dh): %d", INSERT_LOOKBACK_HOURS, len(to_insert))

    if to_insert.empty:
        log.warning("No rows in insert window — skipping Feature Store write.")
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

    log.info("Inserting %d rows into Feature Group '%s'", len(to_insert), fg_name)
    fg.insert(to_insert, write_options={"wait_for_job": True})
    log.info("Done.")


if __name__ == "__main__":
    run()
