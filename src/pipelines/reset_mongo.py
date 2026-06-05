"""
Wipe MongoDB feature store, model registry, and GridFS to free Atlas storage.

Usage:
    python src/pipelines/reset_mongo.py
    python run_pipeline.py reset
"""

import argparse
import logging
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
from dotenv import load_dotenv

from src.utils.mongo_store import clear_mongodb_data

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg_path = os.path.join(_REPO_ROOT, "config", "settings.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(confirm: bool = False):
    cfg = load_config()
    db_name = os.environ.get("MONGODB_DB", cfg.get("mongodb", {}).get("database", "aqi_predictor"))

    if not confirm:
        log.error(
            "Refusing to wipe MongoDB without confirmation. "
            "Re-run with: python run_pipeline.py reset --yes"
        )
        sys.exit(1)

    log.info("Clearing MongoDB database: %s", db_name)
    stats = clear_mongodb_data(cfg)
    log.info(
        "Done — features=%d, registry=%d, gridfs_files=%d, gridfs_chunks=%d",
        stats["features_deleted"],
        stats["registry_deleted"],
        stats["gridfs_files_deleted"],
        stats["gridfs_chunks_deleted"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear MongoDB feature store and models")
    parser.add_argument("--yes", action="store_true", help="Confirm destructive wipe")
    args = parser.parse_args()
    run(confirm=args.yes)
