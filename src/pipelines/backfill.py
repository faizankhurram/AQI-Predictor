"""
Backfill pipeline — run ONCE to populate MongoDB with historical feature data.

Usage:
    python src/pipelines/backfill.py                  # default 90 days
    python src/pipelines/backfill.py --days 180        # extend backfill
    python src/pipelines/backfill.py --csv-only        # save CSV, skip MongoDB
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import time
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
import yaml
from src.data.openmeteo_client import fetch_combined
from src.features.build_features import build_features, drop_incomplete_rows
from src.utils.mongo_store import upsert_features

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BATCH_DAYS = 7   # days per API call to stay well within rate limits
SLEEP_S = 0.3    # polite pause between batches


def load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def date_batches(start: datetime, end: datetime, batch_days: int):
    """Yield (start_str, end_str) pairs covering [start, end] in chunks."""
    cursor = start
    while cursor < end:
        batch_end = min(cursor + timedelta(days=batch_days - 1), end)
        yield cursor.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")
        cursor = batch_end + timedelta(days=1)


def run(backfill_days: int = 90, csv_only: bool = False):
    cfg = load_config()
    lat = cfg["location"]["latitude"]
    lon = cfg["location"]["longitude"]

    end_dt = datetime.utcnow() - timedelta(days=1)   # yesterday (archive available)
    start_dt = end_dt - timedelta(days=backfill_days)

    log.info("Backfilling %d days: %s → %s", backfill_days,
             start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))

    all_frames = []
    batches = list(date_batches(start_dt, end_dt, BATCH_DAYS))
    for i, (s, e) in enumerate(batches, 1):
        log.info("Batch %d/%d: %s → %s", i, len(batches), s, e)
        try:
            raw = fetch_combined(lat, lon, s, e, is_historical=True)
            featured = build_features(raw)
            clean = drop_incomplete_rows(featured)
            all_frames.append(clean)
            log.info("  → %d clean rows", len(clean))
        except Exception as exc:
            log.warning("  Batch failed (%s); skipping.", exc)
        time.sleep(SLEEP_S)

    if not all_frames:
        log.error("No data collected — check API connectivity.")
        sys.exit(1)

    full_df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    log.info("Total rows collected: %d", len(full_df))

    # Always save a local CSV backup
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backfill.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    full_df.to_csv(csv_path, index=False)
    log.info("CSV backup saved → %s", csv_path)

    if csv_only:
        log.info("--csv-only flag set; skipping MongoDB upsert.")
        return

    log.info("Upserting %d rows into MongoDB feature collection...", len(full_df))
    upserted = upsert_features(full_df, cfg)
    log.info("Upsert submitted for %d rows.", upserted)
    log.info("Backfill complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90, help="Days to backfill (default 90)")
    parser.add_argument("--csv-only", action="store_true", help="Save CSV, skip MongoDB")
    args = parser.parse_args()
    run(backfill_days=args.days, csv_only=args.csv_only)
