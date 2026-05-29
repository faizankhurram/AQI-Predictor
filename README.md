# Karachi AQI Predictor

End-to-end **3-day US AQI forecasting** for Karachi, Pakistan — from live environmental data to a deployable model and interactive dashboard.

---

## Overview

Karachi regularly experiences elevated particulate pollution from traffic, industry, and seasonal dust. This project builds an automated pipeline that ingests hourly air-quality and weather data, engineers time-series features, trains regression models on historical patterns, and serves **+24h / +48h / +72h** AQI forecasts through a Streamlit UI and optional REST API.

**Goals**

- Provide residents and planners with short-horizon AQI outlooks (not just current readings).
- Run reliably with minimal manual ops: scheduled ingest, daily retraining, cloud feature store.
- Stay reproducible: versioned config, time-based evaluation, and explainability (SHAP) on the dashboard.

**How it works (high level)**

1. **Ingest** — Open-Meteo air-quality + weather APIs (PM2.5, PM10, NO₂, O₃, wind, humidity, etc.).
2. **Feature store** — Hourly rows in MongoDB (`aqi_hourly_v1`) with calibrated PM2.5, EPA 2024 AQI, lags, 24h pollutant rolling means, cyclic time, wind U/V, and multi-horizon targets.
3. **Train** — Compare Linear, Ridge, Random Forest, and XGBoost; prune correlated features; register the best model in MongoDB GridFS.
4. **Serve** — Streamlit dashboard and FastAPI endpoints read the latest features + registered model to produce forecasts and alerts.

---

## Tech stack

| Layer         | Tools                                                                 |
|---------------|-----------------------------------------------------------------------|
| Data          | [Open-Meteo](https://open-meteo.com/) Air Quality + Forecast APIs       |
| Storage       | MongoDB Atlas (features + model registry / GridFS)                    |
| ML            | scikit-learn, XGBoost; optional TensorFlow MLP                        |
| Orchestration | GitHub Actions (hourly ingest, daily train)                           |
| UI / API      | Streamlit, FastAPI, Plotly                                            |
| Config        | `config/settings.yaml`, `.env`                                        |

---

## Project layout

```
.
├── run_pipeline.py              # feature | train | backfill
├── config/settings.yaml
├── src/
│   ├── dashboard.py             # Streamlit UI
│   ├── data/openmeteo_client.py
│   ├── features/build_features.py
│   ├── models/
│   │   ├── sklearn_trainer.py   # training + metrics
│   │   └── tf_trainer.py        # optional MLP
│   ├── pipelines/               # backfill, feature, training scripts
│   ├── serving/predict.py       # inference + FastAPI app
│   └── utils/mongo_store.py
├── notebooks/eda_quick.ipynb
├── report/report.md             # internship / project write-up
└── .github/workflows/
```

---

## Quick start

### Prerequisites

- Python 3.11+
- [MongoDB Atlas](https://www.mongodb.com/atlas/database) (free tier)
- GitHub account (for CI/CD)

Atlas: create a DB user, allow your IP (or `0.0.0.0/0` for demos), copy the SRV URI into `MONGODB_URI`.

### Setup

```bash
git clone <your-repo-url>
cd "AQI Predictor"

py -3.12 -m venv .venv311
.\.venv311\Scripts\Activate.ps1   # Windows
# source .venv311/bin/activate    # macOS/Linux

pip install -r requirements.txt
cp .env.example .env              # set MONGODB_URI, MONGODB_DB
```

### Pipelines

```bash
# One-time history (90 days → MongoDB or CSV)
python run_pipeline.py backfill --days 90
python run_pipeline.py backfill --days 90 --csv-only   # skip MongoDB

# Train (MongoDB or local CSV)
python run_pipeline.py train
python run_pipeline.py train --csv data/backfill.csv

# Hourly ingest (also run by GitHub Actions)
python run_pipeline.py feature
```

### Dashboard & API

```bash
streamlit run src/dashboard.py
# Sidebar: "Use local model" to skip MongoDB for UI-only demos

uvicorn src.serving.predict:app --reload
# GET /health  /predict  /predict/local
```

### EDA notebook

```powershell
pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name aqi-predictor --display-name "AQI Predictor (.venv311)"
jupyter notebook notebooks/eda_quick.ipynb
```

Figures save to `notebooks/visuals/` (git-ignored).

### Local metrics viewer

```bash
python show_model_metrics.py --detailed    # git-ignored dev script
```

---

## CI/CD

1. Push to GitHub; add secrets `MONGODB_URI` (and optional `MONGODB_DB`).
2. Workflows: **Feature Pipeline (Hourly)** → `python run_pipeline.py feature`; **Training (Daily)** → `python run_pipeline.py train`.
3. Manual test: Actions → Feature Pipeline → Run workflow on `main`.

---

## Environment variables

| Variable      | Description                              |
|---------------|------------------------------------------|
| `MONGODB_URI` | Atlas connection string                  |
| `MONGODB_DB`  | Database name (default: `aqi_predictor`) |

Collections: `aqi_hourly_v1` (unique `timestamp`), `model_registry` + GridFS.

---

## AQI categories (US EPA)

| US AQI  | Category                          |
|---------|-----------------------------------|
| 0–50    | Good                              |
| 51–100  | Moderate                          |
| 101–150 | Unhealthy for Sensitive Groups    |
| 151–200 | Unhealthy                         |
| 201–300 | Very Unhealthy                    |
| 301+    | Hazardous                         |

The dashboard shows an alert banner when current or any forecast AQI exceeds 150.

---
