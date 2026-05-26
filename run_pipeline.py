#!/usr/bin/env python3
"""
Root entry point for scheduled and local pipeline runs.

  python run_pipeline.py feature          # hourly ingest (CI)
  python run_pipeline.py train            # daily training (CI)
  python run_pipeline.py train --csv data/backfill.csv
  python run_pipeline.py backfill --days 90
"""

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
    train_p.add_argument("--with-tf", action="store_true", help="Also train TensorFlow MLP")
    train_p.add_argument("--test-days", type=int, default=14, help="Holdout size in days")

    backfill_p = sub.add_parser("backfill", help="One-time historical load")
    backfill_p.add_argument("--days", type=int, default=90)
    backfill_p.add_argument("--csv-only", action="store_true")

    args = parser.parse_args()

    if args.command == "feature":
        from src.pipelines.feature_pipeline import run
        run()
    elif args.command == "train":
        from src.pipelines.training_pipeline import run
        run(csv_path=args.csv, with_tf=args.with_tf, test_days=args.test_days)
    elif args.command == "backfill":
        from src.pipelines.backfill import run as run_backfill
        run_backfill(backfill_days=args.days, csv_only=args.csv_only)


if __name__ == "__main__":
    main()
