# AQI Predictor — Karachi
## Internship Project Report

**City:** Karachi, Pakistan (24.8607°N, 67.0011°E)  
**Submission Deadline:** 26 May 2026  
**Stack:** Open-Meteo · MongoDB Atlas · scikit-learn · GitHub Actions · Streamlit · FastAPI

---

## 1. Problem Statement

Karachi faces recurring poor air quality from traffic, industry, and seasonal dust. Residents need more than a single current reading—they need a short-horizon outlook to plan outdoor activity and early response. This project delivers an automated **3-day US AQI forecast** (+24h, +48h, +72h) for Karachi using hourly Open-Meteo data, a cloud feature store, daily model retraining, and a public Streamlit dashboard (with optional FastAPI endpoints).

---

## 2. System Architecture

```
Open-Meteo (Air Quality + Weather)
        │
        ▼
run_pipeline.py feature  ──►  MongoDB (aqi_hourly_v1)
        │                              │
   (hourly, GitHub Actions)      run_pipeline.py train
                                       │ (daily, GitHub Actions)
                                       ▼
                                MongoDB GridFS + model_registry
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   FastAPI (predict.py)      Streamlit (dashboard.py)
                          │
                3-day forecast + alerts + SHAP
```

**Data flow.** Open-Meteo supplies PM2.5, PM10, NO₂, O₃, temperature, humidity, wind speed/direction, and precipitation—no API key required. A one-time backfill loads ~90 days of history; an hourly GitHub Action keeps MongoDB current. A daily training job compares four regressors, registers the best model in GridFS, and stores metrics in the registry.

**Serving.** The Streamlit app (deployed on Streamlit Community Cloud) reads the latest features and registered model from MongoDB. FastAPI exposes `/health`, `/predict`, and `/predict/local` for programmatic access.

---

## 3. Tech Stack & CI/CD

**Stack.** Data comes from Open-Meteo (air quality + weather). MongoDB Atlas stores hourly features (`aqi_hourly_v1`) and registered models (`model_registry` + GridFS). Models are built with scikit-learn and XGBoost. The dashboard runs on Streamlit Community Cloud; inference is also exposed via FastAPI. Configuration lives in `config/settings.yaml` and environment secrets (`.env` locally, Streamlit/GitHub secrets in deployment).

**CI/CD.** Two GitHub Actions workflows in `.github/workflows/` keep the pipeline hands-free: the **Feature Pipeline** runs every hour (`python run_pipeline.py feature`) to fetch recent data and upsert MongoDB; the **Training Pipeline** runs daily at 02:00 UTC (`python run_pipeline.py train`) to retrain and register the best model. Both use `MONGODB_URI` from GitHub Secrets and support manual triggers via `workflow_dispatch`.

---

## 4. Methodology

**Feature engineering.** Raw PM2.5 is calibrated (×1.42) for South Asian sensor bias; US AQI is recomputed with 2024 EPA breakpoints. Engineered inputs include 24h rolling means per pollutant, cyclic hour/day/month encodings, wind U/V components, AQI lags and change rates, and interaction indices (smog, dispersion, heat). Targets are future AQI at +24h, +48h, and +72h via backward shift. Incomplete rows from lags/rolls are dropped; volatile columns are winsorized on the train split only.

**Modelling.** Four multi-output regressors (Linear, Ridge, Random Forest, XGBoost) predict all three horizons jointly. Features with pairwise correlation ≥0.85 are pruned, keeping the variable more correlated with the +24h target. Evaluation uses a **time-based holdout** (last 14 days as test)—no random shuffle. The lowest average RMSE model is saved locally and registered in MongoDB.

---

## 5. Results and Discussion

**Model comparison** (fill from `models_artifacts/metrics.json` or `show_model_metrics.py` after training):

| Model             | Avg RMSE | Avg MAE | Avg R² |
|-------------------|----------|---------|--------|
| Linear Regression | _fill_   | _fill_  | _fill_ |
| Ridge             | _fill_   | _fill_  | _fill_ |
| Random Forest     | _fill_   | _fill_  | _fill_ |
| XGBoost           | _fill_   | _fill_  | _fill_ |

The best model is registered in MongoDB GridFS. Optional TensorFlow MLP (128→64→3) is trained only if it beats sklearn on average RMSE.

**EDA highlights** (from `notebooks/eda_quick.ipynb`; images in `notebooks/visuals/`):

- **PM2.5 dominates US AQI (correlation >0.95).**  
  Insert: `07_correlation.png` — correlation heatmap; PM2.5 / `aqi_us` / lag columns cluster tightly.  
  Optional support: `01_calibration_impact.png` — PM2.5 before/after calibration vs recomputed AQI.

  ![PM2.5–AQI correlation](../notebooks/visuals/07_correlation.png)

- **AQI peaks in early morning (06:00–09:00) and evening (18:00–22:00).**  
  Insert: `06_hourly_monthly_pattern.png` — **left panel** (“Average AQI by hour of day”).

  ![Hourly AQI pattern](../notebooks/visuals/06_hourly_monthly_pattern.png)

- **Higher wind speed correlates with lower AQI (dispersion).**  
  Insert: `07_correlation.png` — locate `wind_speed_10m` vs `aqi_us` (and vs `dispersion_index` if present); negative association supports the dispersion story.

- **Winter months (Nov–Feb) show elevated particulates.**  
  Insert: `06_hourly_monthly_pattern.png` — **right panel** (“Average AQI by month”); Nov–Feb bars sit higher than summer months.

**Optional context figures** (not tied to the bullets above, useful for appendix or extra page):

| Figure | File | Use |
|--------|------|-----|
| AQI over backfill window | `04_aqi_timeseries.png` | Shows variability over time |
| Category frequency | `05_aqi_distribution.png` | Hours spent in each EPA band |
| Lag vs +24h target | `09_lag_vs_target_24h.png` | Persistence of AQI forecasting |
| Top features for +24h | `11_feature_target_assoc_24h.png` | Strongest \|r\| with target |

**Dashboard.** The deployed UI shows current AQI, +1d/+2d/+3d forecast cards, a 7-day history chart with EPA zone bands, pollutant snapshot, hazard banner when any forecast exceeds 150, and optional SHAP for the +24h horizon.

**Limitations.** Forecasts rely on Open-Meteo rather than independent ground stations; US AQI is PM2.5-driven; 90-day training may miss full seasonal cycles; Karachi-only coordinates (other cities need config changes only).

---

## 6. Conclusion

<!-- Complete this section yourself. -->
