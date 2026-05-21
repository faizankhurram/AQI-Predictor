# AQI Predictor — Karachi
## Internship Project Report

**City:** Karachi, Pakistan (24.8607°N, 67.0011°E)  
**Submission Deadline:** 26 May 2026  
**Stack:** Open-Meteo · MongoDB Atlas · scikit-learn · GitHub Actions · Streamlit · FastAPI

---

## 1. Problem Statement

Air quality in Karachi regularly reaches hazardous levels due to vehicular emissions, industrial activity, and seasonal dust. Timely 3-day forecasts allow residents and authorities to take preventive action. This project builds an end-to-end, serverless ML pipeline that automatically collects hourly air-quality and weather data, trains forecasting models daily, and surfaces predictions through an interactive web dashboard.

---

## 2. System Architecture

```
Open-Meteo (Air Quality + Weather)
        │
        ▼
feature_pipeline.py  ──►  MongoDB Feature Collection (aqi_hourly_v1)
        │                                │
(hourly, GitHub Actions)          training_pipeline.py
                                         │ (daily, GitHub Actions)
                                         ▼
                                  MongoDB GridFS Model Registry
                                         │
                            ┌────────────┴────────────┐
                            ▼                         ▼
                         FastAPI /predict        Streamlit UI
                                         │
                               3-day AQI forecast + alerts + SHAP
```

---

## 3. Data Sources

| Source | Endpoint | Variables |
|--------|----------|-----------|
| Open-Meteo Air Quality | `air-quality-api.open-meteo.com` | PM2.5, PM10, NO₂, O₃, US AQI |
| Open-Meteo Weather Archive | `archive-api.open-meteo.com` | Temperature, Humidity, Wind speed, Precipitation |
| Open-Meteo Forecast | `api.open-meteo.com` | Same weather variables (live) |

No API key required. Historical archive used for backfill; forecast endpoint used in live pipeline and inference.

---

## 4. Feature Engineering

Each hourly row in the MongoDB feature collection contains:

| Category | Features |
|----------|----------|
| Raw pollutants | `pm2_5`, `pm10`, `no2`, `o3` |
| AQI | `aqi_us` (US EPA scale, from API or computed from PM2.5 breakpoints) |
| Weather | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m` |
| Calendar | `hour`, `day_of_week`, `month` |
| Lag features | `aqi_lag_1h`, `aqi_lag_24h` |
| Change rates | `aqi_change_1h`, `aqi_change_24h` |
| Targets | `aqi_t_plus_24h`, `aqi_t_plus_48h`, `aqi_t_plus_72h` |

**Leakage prevention:** Targets are computed by shifting future AQI values. Time-based split (last 14 days = test) is used — no random shuffling.

---

## 5. Models and Results

Two models trained on the multi-output regression task (predict AQI at +24h, +48h, +72h simultaneously):

| Model | Avg RMSE | Avg MAE | Avg R² |
|-------|----------|---------|--------|
| Ridge Regression | _fill after training_ | _fill_ | _fill_ |
| Random Forest | _fill after training_ | _fill_ | _fill_ |

> Best model registered in MongoDB GridFS model registry. See `models_artifacts/metrics.json` for full per-horizon breakdown.

### Optional: TensorFlow MLP
A 2-layer MLP (128 → 64 → 3 outputs) was also trained. It can be registered to the MongoDB model registry if its average RMSE beats the sklearn best model.

---

## 6. CI/CD Pipeline

| Workflow | Schedule | Trigger |
|----------|----------|---------|
| `feature_pipeline.yml` | Every hour (`0 * * * *`) | `workflow_dispatch` available |
| `training_pipeline.yml` | Daily at 02:00 UTC (`0 2 * * *`) | `workflow_dispatch` available |

Secrets required: `MONGODB_URI` (and optional `MONGODB_DB`) set in GitHub → Settings → Secrets.

---

## 7. Dashboard Features

- **Current conditions card** — live US AQI + label
- **3-day forecast cards** — +1d, +2d, +3d predicted AQI with colour-coded labels
- **Historical + forecast chart** — 7-day trend + forecast overlay
- **Pollutant snapshot table** — PM2.5, PM10, NO₂, O₃, Temperature, Humidity, Wind
- **Hazard alerts** — persistent banner when current or any forecast AQI ≥ 150
- **SHAP explainability panel** — top 5 feature contributions for the latest prediction

---

## 8. Exploratory Data Analysis

<!-- Insert screenshots from notebooks/eda_quick.ipynb -->

**AQI time series:**  
![AQI Time Series](eda_aqi_timeseries.png)

**Hourly pattern:**  
![Hourly Pattern](eda_hourly_pattern.png)

**Correlation matrix:**  
![Correlation](eda_correlation.png)

**AQI distribution:**  
![Distribution](eda_aqi_distribution.png)

Key findings:
- PM2.5 is the strongest predictor of US AQI (correlation > 0.95).
- AQI peaks during early morning (06:00–09:00) and evening (18:00–22:00) hours.
- Wind speed shows moderate negative correlation — higher winds disperse pollutants.
- Winter months (Nov–Feb) show significantly elevated PM2.5.

---

## 9. Hazard Alert Thresholds

| US AQI | Category | Action |
|--------|----------|--------|
| 0–50 | Good | None |
| 51–100 | Moderate | Unusually sensitive people should consider limiting outdoor activity |
| 101–150 | Unhealthy for Sensitive Groups | Sensitive individuals should limit prolonged outdoor exertion |
| 151–200 | Unhealthy | Everyone may begin to experience health effects |
| 201–300 | Very Unhealthy | Health alert: everyone may experience more serious effects |
| 301+ | Hazardous | Health warnings of emergency conditions |

Dashboard displays a coloured banner whenever current or any forecast day crosses 150.

---

## 10. Limitations

1. **Single pollutant emphasis:** US AQI is primarily driven by PM2.5; NO₂ and O₃ are available but rarely dominant in Karachi's profile — model may underweight them.
2. **Open-Meteo coverage:** Karachi's air-quality station coverage can have sporadic gaps; forward-fill of up to 6 hours is applied.
3. **Single city:** The pipeline is tuned for Karachi coordinates. Extending to other cities requires only config changes.
4. **Short backfill:** 90-day training window is sufficient for seasonal patterns but a full year would improve winter/summer model discrimination.
5. **No real-time ground truth:** The model targets are derived from the same Open-Meteo API rather than independent monitoring stations — an alternative ground-truth source (e.g., Pakistan EPA) would strengthen evaluation.

---

## 11. Reproduction Instructions

See `README.md` for step-by-step setup. Summary:

```bash
# 1. Clone and set up
git clone <repo-url>
cd AQI-Predictor
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # fill in MongoDB credentials

# 2. Backfill 90 days
python src/pipelines/backfill.py --days 90

# 3. Train models
python src/pipelines/training_pipeline.py

# 4. Launch dashboard
streamlit run src/app/streamlit_app.py

# 5. Launch API (optional)
uvicorn src.serving.api:app --reload
```
