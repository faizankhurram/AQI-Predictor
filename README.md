# Karachi AQI Predictor

End-to-end, serverless 3-day Air Quality Index forecasting for **Karachi, Pakistan**.

Built with Open-Meteo · MongoDB Atlas · scikit-learn · GitHub Actions · Streamlit · FastAPI

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A free [MongoDB Atlas](https://www.mongodb.com/atlas/database) cluster
- A GitHub account (for CI/CD)

MongoDB Atlas setup:
- Create a database user with read/write access.
- Add your current IP address for local development.
- For GitHub Actions, allow runner access in **Network Access** (for a student/demo project, `0.0.0.0/0` is simplest; use a restricted rule if your organization provides one).
- Copy the SRV connection string into `MONGODB_URI`.

### 2. Setup

```bash
git clone <your-repo-url>
cd "AQI Predictor"

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
# Edit .env → set MONGODB_URI and MONGODB_DB
```

### 3. Backfill historical data (run once)

```bash
# Fetch 90 days of Open-Meteo data and upsert into MongoDB
python src/pipelines/backfill.py --days 90

# Or save to local CSV only (skip MongoDB, useful for local dev)
python src/pipelines/backfill.py --days 90 --csv-only
```

### 4. Train models

```bash
# From MongoDB feature store (requires credentials)
python src/pipelines/training_pipeline.py

# From local CSV backup (no MongoDB required)
python src/pipelines/training_pipeline.py --csv data/backfill.csv

# Also train optional TensorFlow MLP
python src/pipelines/training_pipeline.py --csv data/backfill.csv --with-tf
```

### 5. Launch the dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Toggle "Use local model" in the sidebar to avoid needing MongoDB credentials for the UI.

### 6. Launch the API (optional)

```bash
uvicorn src.serving.api:app --reload
# Endpoints:
#   GET /health
#   GET /predict        (uses MongoDB features + model registry)
#   GET /predict/local  (uses local model + live Open-Meteo)
```

### 7. Run EDA notebook

```bash
jupyter notebook notebooks/eda_quick.ipynb
# Requires data/backfill.csv to exist first
```

---

## Project Layout

```
.
├── .github/workflows/
│   ├── feature_pipeline.yml   # Runs every hour
│   └── training_pipeline.yml  # Runs daily at 02:00 UTC
├── config/settings.yaml        # Karachi lat/lon, thresholds, MongoDB names
├── data/                       # Local CSV backup (git-ignored)
├── models_artifacts/           # Saved .pkl + metrics.json (git-ignored)
├── notebooks/eda_quick.ipynb
├── report/report.md
├── requirements.txt
├── src/
│   ├── data/openmeteo_client.py
│   ├── features/build_features.py
│   ├── pipelines/
│   │   ├── feature_pipeline.py
│   │   ├── backfill.py
│   │   └── training_pipeline.py
│   ├── models/
│   │   ├── sklearn_trainer.py
│   │   ├── tf_trainer.py
│   │   └── metrics.py
│   ├── serving/
│   │   ├── predict.py
│   │   └── api.py
│   └── app/streamlit_app.py
└── .env.example
```

---

## CI/CD Setup

1. Push this repo to GitHub.
2. Go to **Settings → Secrets and variables → Actions**.
3. Add secrets:
   - `MONGODB_URI` — MongoDB Atlas connection string
   - Optional: `MONGODB_DB` — database name (defaults to `aqi_predictor`)

   Workflows run `pip install -e .` then `python run_feature_pipeline.py` from the repo root.
   Commit `pyproject.toml`, `run_feature_pipeline.py`, and the full `src/` tree (folder must be **`src/`**, not `scr/`).

   **Important:** If a run fails, use **Actions → Feature Pipeline → Run workflow** on branch `main`.
   Do **not** use **Re-run failed jobs** — that re-executes the old commit (before `src/data/` existed) and keeps the broken command `python src/pipelines/feature_pipeline.py`.
4. The hourly and daily workflows will start automatically on the schedule.
5. To test immediately: go to **Actions → Feature Pipeline (Hourly) → Run workflow**.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | Database name (default: `aqi_predictor`) |

Copy `.env.example` to `.env` and fill in the values. Never commit `.env` to git.

MongoDB collections are created automatically:
- `aqi_hourly_v1` for feature rows, unique on `timestamp`
- `model_registry` + GridFS (`fs.files`, `fs.chunks`) for model artifacts

---

## Hazard Alert Thresholds

| US AQI | Category |
|--------|----------|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

The dashboard shows a coloured banner when current or any forecast AQI exceeds 150.

---

## MongoDB Sanity Check

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from src.utils.mongo_store import get_database; print(get_database().name)"
```
