"""
Streamlit AQI Predictor Dashboard — Karachi
Run: streamlit run src/app/streamlit_app.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.serving.predict import predict, aqi_label
from src.features.build_features import get_feature_columns

log = logging.getLogger(__name__)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)

AQI_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}

AQI_BG = {
    "Good": "#d4f7d4",
    "Moderate": "#fffacd",
    "Unhealthy for Sensitive Groups": "#ffe4b5",
    "Unhealthy": "#ffcccc",
    "Very Unhealthy": "#e6ccff",
    "Hazardous": "#ffb3c1",
}


def aqi_color(label: str) -> str:
    return AQI_COLORS.get(label, "#cccccc")


def aqi_bg(label: str) -> str:
    return AQI_BG.get(label, "#f0f0f0")


def load_historical_csv() -> pd.DataFrame | None:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backfill.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df.sort_values("timestamp")
    return None


def render_alert(forecasts: list):
    """Show a persistent banner if any forecast horizon is in hazardous range."""
    worst_aqi = max(f["aqi_us"] for f in forecasts)
    worst_label = aqi_label(worst_aqi)
    if worst_aqi > 150:
        color = aqi_color(worst_label)
        bg = aqi_bg(worst_label)
        st.markdown(
            f"""
            <div style="background-color:{bg}; border-left: 6px solid {color};
                        padding: 12px 20px; border-radius: 6px; margin-bottom: 16px;">
                <strong>⚠️ Air Quality Alert — Karachi</strong><br>
                Forecast peak: <strong>{worst_aqi:.0f} US AQI</strong>
                — <span style="color:{color}; font-weight:bold;">{worst_label}</span>.<br>
                Sensitive individuals should avoid prolonged outdoor exposure.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_cards(forecasts: list, current_aqi: float | None):
    cols = st.columns(4)
    if current_aqi is not None:
        label = aqi_label(current_aqi)
        cols[0].markdown(
            f"""
            <div style="text-align:center; background:{aqi_bg(label)};
                        border-radius:10px; padding:16px;">
                <div style="font-size:0.85rem; color:#555;">Now</div>
                <div style="font-size:2rem; font-weight:700;">{current_aqi:.0f}</div>
                <div style="font-size:0.8rem; color:{aqi_color(label)}; font-weight:600;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    for col, fc in zip(cols[1:], forecasts):
        label = fc["label"]
        col.markdown(
            f"""
            <div style="text-align:center; background:{aqi_bg(label)};
                        border-radius:10px; padding:16px;">
                <div style="font-size:0.85rem; color:#555;">+{fc['horizon_h']//24}d</div>
                <div style="font-size:2rem; font-weight:700;">{fc['aqi_us']:.0f}</div>
                <div style="font-size:0.8rem; color:{aqi_color(label)}; font-weight:600;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_chart(hist_df: pd.DataFrame, forecasts: list, last_ts):
    """Plot last 7 days of historical AQI + 3 forecast points."""
    cutoff = pd.Timestamp(last_ts) - timedelta(days=7) if last_ts else hist_df["timestamp"].max() - timedelta(days=7)
    recent = hist_df[hist_df["timestamp"] >= cutoff].copy()

    fig = go.Figure()

    # Historical trace
    fig.add_trace(go.Scatter(
        x=recent["timestamp"],
        y=recent["aqi_us"],
        mode="lines",
        name="Historical AQI",
        line=dict(color="#1f77b4", width=2),
    ))

    # Forecast points
    if last_ts:
        base_ts = pd.Timestamp(last_ts)
        fc_times = [base_ts + timedelta(hours=f["horizon_h"]) for f in forecasts]
        fc_vals = [f["aqi_us"] for f in forecasts]
        fig.add_trace(go.Scatter(
            x=fc_times,
            y=fc_vals,
            mode="markers+lines",
            name="Forecast",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        ))

    # AQI band reference lines
    for threshold, label, color in [
        (50, "Good", "#00e400"),
        (100, "Moderate", "#ffff00"),
        (150, "Sensitive", "#ff7e00"),
        (200, "Unhealthy", "#ff0000"),
    ]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font_size=10)

    fig.update_layout(
        title="Historical AQI (7 days) + 3-Day Forecast",
        xaxis_title="Time",
        yaxis_title="US AQI",
        legend=dict(orientation="h", y=1.02, x=0),
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap(feature_row: dict):
    """SHAP explainability panel — loads model from disk, explains latest prediction."""
    import joblib

    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models_artifacts", "best_model.pkl")
    if not os.path.exists(model_path):
        st.info("SHAP not available — run the training pipeline first to generate a local model.")
        return

    try:
        import shap
        pipeline = joblib.load(model_path)
        feature_cols = get_feature_columns()
        X = pd.DataFrame([feature_row])[feature_cols].values

        # Get the underlying estimator (MultiOutputRegressor wrapping RF or Ridge)
        estimator = pipeline.named_steps.get("model", pipeline)

        # For MultiOutputRegressor, explain the first sub-estimator (+24h)
        inner = estimator.estimators_[0] if hasattr(estimator, "estimators_") else estimator

        explainer = shap.TreeExplainer(inner) if hasattr(inner, "feature_importances_") \
            else shap.LinearExplainer(inner, X)

        if "scaler" in pipeline.named_steps:
            X_scaled = pipeline.named_steps["scaler"].transform(X)
        else:
            X_scaled = X

        shap_values = explainer(X_scaled)
        top_n = 5
        importances = pd.DataFrame({
            "Feature": feature_cols,
            "SHAP Value": np.abs(shap_values.values[0]),
        }).sort_values("SHAP Value", ascending=False).head(top_n)

        st.subheader("Top Feature Contributions (latest prediction, +24h horizon)")
        fig = go.Figure(go.Bar(
            x=importances["SHAP Value"],
            y=importances["Feature"],
            orientation="h",
            marker_color="#636efa",
        ))
        fig.update_layout(
            xaxis_title="|SHAP value|",
            yaxis_title="Feature",
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"SHAP computation skipped: {exc}")


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.title("🌫️ Karachi AQI Forecaster")
    st.caption("3-day Air Quality Index prediction · Powered by Open-Meteo + Hopsworks + scikit-learn")

    # Sidebar controls
    st.sidebar.header("Settings")
    use_local = st.sidebar.toggle("Use local model (no Hopsworks)", value=True)
    show_shap = st.sidebar.toggle("Show SHAP explanation", value=True)
    refresh = st.sidebar.button("🔄 Refresh prediction")

    if "prediction" not in st.session_state or refresh:
        with st.spinner("Running inference..."):
            try:
                st.session_state["prediction"] = predict(local=use_local)
                st.session_state["pred_error"] = None
            except Exception as exc:
                st.session_state["pred_error"] = str(exc)

    if st.session_state.get("pred_error"):
        st.error(f"Prediction failed: {st.session_state['pred_error']}")
        st.info("Make sure you have run the training pipeline at least once (`python src/pipelines/training_pipeline.py --csv data/backfill.csv`) and the backfill CSV exists.")
        return

    result = st.session_state["prediction"]
    forecasts = result["forecasts"]
    current_aqi = result.get("latest_actual")
    last_ts = result.get("latest_timestamp")

    # Alert banner
    render_alert(forecasts)

    # Current + forecast cards
    st.subheader("Current Conditions & 3-Day Forecast")
    render_forecast_cards(forecasts, current_aqi)

    st.divider()

    # Historical chart
    hist_df = load_historical_csv()
    if hist_df is not None and "aqi_us" in hist_df.columns:
        render_history_chart(hist_df, forecasts, last_ts)
    else:
        st.info("No historical data found. Run `python src/pipelines/backfill.py --csv-only` to generate the local CSV.")

    st.divider()

    # Pollutant table
    feature_row = result.get("feature_row", {})
    if feature_row:
        st.subheader("Current Pollutant Snapshot")
        pollutants = {
            "PM2.5 (µg/m³)": feature_row.get("pm2_5"),
            "PM10 (µg/m³)": feature_row.get("pm10"),
            "NO₂ (µg/m³)": feature_row.get("no2"),
            "O₃ (µg/m³)": feature_row.get("o3"),
            "Temperature (°C)": feature_row.get("temperature_2m"),
            "Humidity (%)": feature_row.get("relative_humidity_2m"),
            "Wind Speed (km/h)": feature_row.get("wind_speed_10m"),
        }
        table_df = pd.DataFrame(
            [(k, f"{v:.1f}" if v is not None else "N/A") for k, v in pollutants.items()],
            columns=["Metric", "Value"],
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.divider()

    # SHAP panel
    if show_shap and feature_row:
        with st.expander("🔍 Feature Importance (SHAP)", expanded=False):
            render_shap(feature_row)

    # Footer
    generated_at = result.get("generated_at", "")
    st.caption(f"Prediction generated at {generated_at} UTC")


if __name__ == "__main__":
    main()
