# Karachi AQI Predictor

End-to-end, serverless 3-day Air Quality Index forecasting for **Karachi, Pakistan**.

Built with Open-Meteo · Hopsworks Feature Store + Model Registry · scikit-learn · GitHub Actions · Streamlit · FastAPI

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A free [Hopsworks Serverless](https://app.hopsworks.ai) account (create project `aqi-karachi`, generate API key)
- A GitHub account (for CI/CD)

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

# If Hopsworks fails on Windows due pyjks/twofish build issues,
# use the workaround section below.

# Configure credentials
cp .env.example .env
# Edit .env → set HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, and HOPSWORKS_HOST
```

### 3. Backfill historical data (run once)

```bash
# Fetch 90 days of Open-Meteo data and push to Hopsworks Feature Group
python src/pipelines/backfill.py --days 90

# Or save to local CSV only (skip Hopsworks, useful for local dev)
python src/pipelines/backfill.py --days 90 --csv-only
```

### 4. Train models

```bash
# From Hopsworks Feature Group (requires credentials)
python src/pipelines/training_pipeline.py

# From local CSV backup (no Hopsworks required)
python src/pipelines/training_pipeline.py --csv data/backfill.csv

# Also train optional TensorFlow MLP
python src/pipelines/training_pipeline.py --csv data/backfill.csv --with-tf
```

### 5. Launch the dashboard

```bash
streamlit run src/app/streamlit_app.py
```

Toggle "Use local model" in the sidebar to avoid needing Hopsworks credentials for the UI.

### 6. Launch the API (optional)

```bash
uvicorn src.serving.api:app --reload
# Endpoints:
#   GET /health
#   GET /predict        (uses Hopsworks FG + registry)
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
├── config/settings.yaml        # Karachi lat/lon, thresholds, FG names
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
   - `HOPSWORKS_API_KEY` — your Hopsworks API key
   - `HOPSWORKS_PROJECT` — your Hopsworks project name (e.g. `aqi-karachi`)

   Workflows run `pip install -e .` then `python run_feature_pipeline.py` from the repo root.
   Commit `pyproject.toml`, `run_feature_pipeline.py`, and the full `src/` tree (folder must be **`src/`**, not `scr/`).
4. The hourly and daily workflows will start automatically on the schedule.
5. To test immediately: go to **Actions → Feature Pipeline (Hourly) → Run workflow**.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HOPSWORKS_API_KEY` | Hopsworks Serverless API key |
| `HOPSWORKS_PROJECT` | Hopsworks project name |
| `HOPSWORKS_HOST` | API host (`eu-west.cloud.hopsworks.ai`; do not use `c.app.hopsworks.ai`) |

Copy `.env.example` to `.env` and fill in the values. Never commit `.env` to git.

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

## Hopsworks install (Windows / Serverless)

Serverless backend is **4.7.x** — use matching client and cloud API host:

```bash
pip install -r requirements.txt
pip install "hopsworks==4.7.5"
```

Set in `.env`:

```env
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
```

**Do not** use `pip install hopsworks==3.7.0 --no-deps` / `hsfs --no-deps` — that leaves a broken `hsfs` package (no `connection` module).

Sanity check:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from hsfs import connection; import hopsworks; print('OK')"
```

If `twofish` build fails on older setups, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with C++ workload, then retry `pip install hopsworks==4.7.5`.
