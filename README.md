# Karachi AQI Predictor

End-to-end **3-day US AQI forecasting** for Karachi, Pakistan — from live environmental data to a deployable model and interactive dashboard.

Deployed Link: https://faizan-aqi-predictor-karachi.streamlit.app

---

## Overview

Karachi regularly experiences elevated particulate pollution from traffic, industry, and seasonal dust. This project builds an automated pipeline that ingests hourly air-quality and weather data, engineers time-series features, trains regression models on historical patterns, and serves a **3-day (96-hour)** AQI forecast through a Streamlit UI and optional REST API.

**Modelling approach.** The model is trained to predict the **next hour's** US AQI (`aqi_us.shift(-1)`). Because AQI is highly autocorrelated, this single-step target is learned with high accuracy (R² ≈ 0.9 on real Karachi data). Multi-day outlooks (+24h / +48h / +72h) are then produced by **iterative (recursive) forecasting** — predicting one hour ahead, feeding the prediction back in, and repeating 96 times. This is far more accurate than predicting 1–3 days ahead directly.

**Goals**

- Provide residents and planners with short-horizon AQI outlooks (not just current readings).
- Run reliably with minimal manual ops: scheduled ingest, daily retraining, cloud feature store.
- Stay reproducible: versioned config, time-based evaluation, and explainability (SHAP) on the dashboard.

**How it works (high level)**

1. **Ingest** — Open-Meteo air-quality + weather APIs (PM2.5, PM10, CO, CO₂, NO₂, SO₂, O₃, dust, UV, temperature, humidity, wind, cloud cover).
2. **Clean & engineer** — Gap-fill (ffill/interpolate), IQR outlier capping, then cyclic hour, wind U/V components, 36h rolling means + 72h rolling stds, AQI change rate, PM2.5/PM10 ratio, rush-hour and precipitation-likelihood features.
3. **Feature store** — Hourly engineered rows in MongoDB (`aqi_hourly_v1`), keyed by `timestamp`.
4. **Train** — Compare Random Forest, Gradient Boosting, XGBoost, SVR, and Ridge (MinMaxScaler, time-ordered 80/20 split); select the best by R², retrain on the full series, and register `{model, scaler, features}` in MongoDB GridFS.
5. **Serve** — Streamlit dashboard and FastAPI endpoints run the iterative 96-hour forecast and surface forecasts, alerts, and SHAP explanations.

---

## Tech stack

| Layer         | Tools                                                                 |
|---------------|-----------------------------------------------------------------------|
| Data          | [Open-Meteo](https://open-meteo.com/) Air Quality + Forecast APIs       |
| Storage       | MongoDB Atlas (features + model registry / GridFS)                    |
| ML            | scikit-learn (RF, GBR, SVR, Ridge), XGBoost; SHAP explainability      |
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
│   ├── models/sklearn_trainer.py
│   ├── pipelines/
│   ├── serving/predict.py
│   └── utils/mongo_store.py
├── notebooks/eda_quick.ipynb
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
# One-time history (365 days → MongoDB or CSV)
python run_pipeline.py backfill --days 365
python run_pipeline.py backfill --days 365 --csv-only   # skip MongoDB

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
python show_model_metrics.py --local
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
