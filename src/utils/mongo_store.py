"""MongoDB-backed feature store and lightweight model registry."""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Any

import gridfs
import joblib
import pandas as pd
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database


DEFAULT_DB_NAME = "aqi_predictor"
DEFAULT_FEATURE_COLLECTION = "aqi_hourly_v1"
DEFAULT_MODEL_COLLECTION = "model_registry"
DEFAULT_MODEL_NAME = "aqi_forecaster"


def _mongo_uri() -> str:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is required for MongoDB feature/model storage.")
    return uri


def get_database(db_name: str | None = None) -> Database:
    """Create a MongoDB database handle from environment variables."""
    client = MongoClient(_mongo_uri(), serverSelectionTimeoutMS=30000)
    return client[db_name or os.environ.get("MONGODB_DB", DEFAULT_DB_NAME)]


def _collection_name(cfg: dict | None, key: str, default: str) -> str:
    if cfg:
        return cfg.get("mongodb", {}).get(key, default)
    return default


def get_feature_collection(cfg: dict | None = None) -> Collection:
    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    collection = db[_collection_name(cfg, "feature_collection", DEFAULT_FEATURE_COLLECTION)]
    collection.create_index("timestamp", unique=True)
    collection.create_index("date")
    return collection


def get_model_collection(cfg: dict | None = None) -> Collection:
    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    collection = db[_collection_name(cfg, "model_collection", DEFAULT_MODEL_COLLECTION)]
    collection.create_index([("name", 1), ("created_at", -1)])
    return collection


def _to_mongo_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({key: _to_mongo_value(value) for key, value in row.items()})
    return records


def upsert_features(df: pd.DataFrame, cfg: dict | None = None) -> int:
    """Upsert feature rows by timestamp and return the number of rows submitted."""
    if df.empty:
        return 0

    collection = get_feature_collection(cfg)
    operations = []
    for record in dataframe_to_records(df):
        timestamp = record.get("timestamp")
        if timestamp is None:
            continue
        # Use $set so fields added by later pipeline runs are preserved,
        # rather than replacing the whole document (ReplaceOne would wipe
        # any fields not present in the current batch).
        operations.append(UpdateOne(
            {"timestamp": timestamp},
            {"$set": record},
            upsert=True,
        ))

    if not operations:
        return 0
    collection.bulk_write(operations, ordered=False)
    return len(operations)


def delete_feature_rows_after(timestamp: pd.Timestamp | datetime, cfg: dict | None = None) -> int:
    """Delete feature rows later than the provided local-naive timestamp."""
    collection = get_feature_collection(cfg)
    result = collection.delete_many({"timestamp": {"$gt": _to_mongo_value(timestamp)}})
    return result.deleted_count


def read_features(cfg: dict | None = None) -> pd.DataFrame:
    """Read all feature rows from MongoDB as a timestamp-sorted DataFrame."""
    collection = get_feature_collection(cfg)
    rows = list(collection.find({}, {"_id": 0}).sort("timestamp", 1))
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_model_artifact(
    *,
    name: str,
    model_path: str,
    metrics_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    cfg: dict | None = None,
) -> dict[str, Any]:
    """Store a model artifact in GridFS and metadata in a registry collection."""
    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    fs = gridfs.GridFS(db)
    collection = db[_collection_name(cfg, "model_collection", DEFAULT_MODEL_COLLECTION)]
    collection.create_index([("name", 1), ("created_at", -1)])

    with open(model_path, "rb") as model_file:
        file_id = fs.put(
            model_file,
            filename=os.path.basename(model_path),
            content_type="application/octet-stream",
            metadata={"model_name": name},
        )

    metrics = None
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    document = {
        "name": name,
        "file_id": file_id,
        "filename": os.path.basename(model_path),
        "metrics": metrics,
        "metadata": metadata or {},
        "created_at": datetime.utcnow(),
    }
    result = collection.insert_one(document)
    document["_id"] = result.inserted_id
    return document


def load_latest_model(name: str = DEFAULT_MODEL_NAME, cfg: dict | None = None):
    """Load the newest registered model artifact from MongoDB GridFS."""
    collection = get_model_collection(cfg)
    document = collection.find_one({"name": name}, sort=[("created_at", -1)])
    if not document:
        raise FileNotFoundError(f"No MongoDB model artifact found for '{name}'.")

    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    fs = gridfs.GridFS(db)
    grid_out = fs.get(document["file_id"])
    return joblib.load(io.BytesIO(grid_out.read()))
