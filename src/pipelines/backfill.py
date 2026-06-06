"""Historical backfill into MongoDB and data/backfill.csv."""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import yaml
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.openmeteo_client import fetch_combined
from src.features.build_features import build_features, drop_incomplete_features
from src.utils.mongo_store import upsert_features

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BATCH_DAYS = 7
SLEEP_S = 0.3


def load_config() -> dict:
    cfg_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_batches(start: datetime, end: datetime, batch_days: int):
    cursor = start
    while cursor < end:
        batch_end = min(cursor + timedelta(days=batch_days - 1), end)
        yield cursor.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")
        cursor = batch_end + timedelta(days=1)


def run(backfill_days: int = 365, csv_only: bool = False):
    cfg = load_config()
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]

    end_dt = datetime.utcnow() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=backfill_days)
    log.info("Backfilling %d days: %s → %s", backfill_days, start_dt.date(), end_dt.date())

    raw_frames = []
    batches = list(date_batches(start_dt, end_dt, BATCH_DAYS))
    for i, (s, e) in enumerate(batches, 1):
        log.info("Batch %d/%d: %s → %s", i, len(batches), s, e)
        try:
            raw = fetch_combined(lat, lon, s, e, is_historical=True)
            raw_frames.append(raw)
            log.info("  → %d raw rows", len(raw))
        except Exception as exc:
            log.warning("  Batch failed (%s); skipping.", exc)
        time.sleep(SLEEP_S)

    if not raw_frames:
        log.error("No data collected.")
        sys.exit(1)

    raw_all = (
        pd.concat(raw_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    featured = build_features(raw_all)
    full_df = drop_incomplete_features(featured).drop_duplicates(subset=["timestamp"])
    log.info("Feature rows: %d", len(full_df))

    csv_path = os.path.join(_REPO_ROOT, "data", "backfill.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    full_df.to_csv(csv_path, index=False)
    log.info("CSV saved → %s", csv_path)

    if csv_only:
        return

    upserted = upsert_features(full_df, cfg)
    log.info("MongoDB upsert submitted for %d rows.", upserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    run(backfill_days=args.days, csv_only=args.csv_only)
