"""MongoDB feature store and model registry."""

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
        raise RuntimeError("MONGODB_URI is required.")
    return uri


def get_database(db_name: str | None = None) -> Database:
    timeout_raw = os.environ.get("MONGODB_TIMEOUT_MS")
    if timeout_raw:
        timeout_ms = int(timeout_raw)
        client = MongoClient(
            _mongo_uri(),
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
        )
    else:
        client = MongoClient(_mongo_uri())
    return client[db_name or os.environ.get("MONGODB_DB", DEFAULT_DB_NAME)]


def _collection_name(cfg: dict | None, key: str, default: str) -> str:
    return cfg.get("mongodb", {}).get(key, default) if cfg else default


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
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def dataframe_to_ingest_records(
    df: pd.DataFrame,
    *,
    omit_null_targets: bool = True,
    target_columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if target_columns is None:
        target_columns = ("target",)

    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        record = {key: _to_mongo_value(value) for key, value in row.items()}
        if omit_null_targets:
            for col in target_columns:
                if col in record and record[col] is None:
                    del record[col]
        records.append(record)
    return records


def upsert_features(df: pd.DataFrame, cfg: dict | None = None) -> int:
    if df.empty:
        return 0

    collection = get_feature_collection(cfg)
    operations = []
    for record in dataframe_to_ingest_records(df, omit_null_targets=True):
        timestamp = record.get("timestamp")
        if timestamp is None:
            continue
        operations.append(UpdateOne({"timestamp": timestamp}, {"$set": record}, upsert=True))

    if not operations:
        return 0
    collection.bulk_write(operations, ordered=False)
    return len(operations)


def delete_feature_rows_after(timestamp: pd.Timestamp | datetime, cfg: dict | None = None) -> int:
    collection = get_feature_collection(cfg)
    return collection.delete_many({"timestamp": {"$gt": _to_mongo_value(timestamp)}}).deleted_count


def read_features(cfg: dict | None = None) -> pd.DataFrame:
    collection = get_feature_collection(cfg)
    rows = list(collection.find({}, {"_id": 0}).sort("timestamp", 1))
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_features_since(since: pd.Timestamp | datetime, cfg: dict | None = None) -> pd.DataFrame:
    collection = get_feature_collection(cfg)
    rows = list(
        collection.find({"timestamp": {"$gte": _to_mongo_value(since)}}, {"_id": 0}).sort("timestamp", 1)
    )
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def read_latest_feature_row(cfg: dict | None = None) -> pd.DataFrame:
    collection = get_feature_collection(cfg)
    rows = list(collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(1))
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
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    result = collection.insert_one(document)
    document["_id"] = result.inserted_id
    return document


def get_latest_model_document(name: str = DEFAULT_MODEL_NAME, cfg: dict | None = None) -> dict:
    collection = get_model_collection(cfg)
    document = collection.find_one({"name": name}, sort=[("created_at", -1)])
    if not document:
        raise FileNotFoundError(f"No MongoDB model artifact found for '{name}'.")
    return document


def load_latest_model(name: str = DEFAULT_MODEL_NAME, cfg: dict | None = None):
    document = get_latest_model_document(name, cfg)
    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    grid_out = gridfs.GridFS(db).get(document["file_id"])
    return joblib.load(io.BytesIO(grid_out.read()))


def clear_mongodb_data(cfg: dict | None = None) -> dict[str, int]:
    db = get_database(_collection_name(cfg, "database", DEFAULT_DB_NAME))
    feature_name = _collection_name(cfg, "feature_collection", DEFAULT_FEATURE_COLLECTION)
    model_name = _collection_name(cfg, "model_collection", DEFAULT_MODEL_COLLECTION)

    feature_result = db[feature_name].delete_many({})
    registry_result = db[model_name].delete_many({})
    chunks_result = db["fs.chunks"].delete_many({})
    files_result = db["fs.files"].delete_many({})

    return {
        "database": db.name,
        "features_deleted": feature_result.deleted_count,
        "registry_deleted": registry_result.deleted_count,
        "gridfs_files_deleted": files_result.deleted_count,
        "gridfs_chunks_deleted": chunks_result.deleted_count,
    }
