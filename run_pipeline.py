#!/usr/bin/env python3
"""Entry point for pipeline commands."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="AQI Predictor pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("feature", help="Hourly Open-Meteo ingest to MongoDB")
    train_p = sub.add_parser("train", help="Train models and register best in MongoDB")
    train_p.add_argument("--csv", type=str, default=None, help="Local CSV instead of MongoDB")

    backfill_p = sub.add_parser("backfill", help="One-time historical load")
    backfill_p.add_argument("--days", type=int, default=365)
    backfill_p.add_argument("--csv-only", action="store_true")

    reset_p = sub.add_parser("reset", help="Wipe MongoDB features, registry, and GridFS")
    reset_p.add_argument("--yes", action="store_true")

    args = parser.parse_args()

    if args.command == "feature":
        from src.pipelines.feature_pipeline import run
        run()
    elif args.command == "train":
        from src.pipelines.training_pipeline import run
        run(csv_path=args.csv)
    elif args.command == "backfill":
        from src.pipelines.backfill import run as run_backfill
        run_backfill(backfill_days=args.days, csv_only=args.csv_only)
    elif args.command == "reset":
        from src.pipelines.reset_mongo import run as run_reset
        run_reset(confirm=args.yes)


if __name__ == "__main__":
    main()
