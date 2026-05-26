"""
Streamlit AQI Predictor Dashboard — Karachi
Run: streamlit run src/dashboard.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.serving.predict import predict, aqi_label
from src.features.build_features import load_training_feature_columns
from src.utils.mongo_store import read_features

log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Karachi AQI Forecast",
    page_icon="🌫️",
    layout="wide",
)

# ── AQI colour maps ──────────────────────────────────────────────────────────
AQI_COLORS = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
}
AQI_BG = {
    "Good":                           "#d4f7d4",
    "Moderate":                       "#fffacd",
    "Unhealthy for Sensitive Groups": "#ffe4b5",
    "Unhealthy":                      "#ffcccc",
    "Very Unhealthy":                 "#e6ccff",
    "Hazardous":                      "#ffb3c1",
}
# (y0, y1, fill-colour, label)
AQI_BANDS = [
    (0,   50,  "#00e400", "Good"),
    (50,  100, "#ffff00", "Moderate"),
    (100, 150, "#ff7e00", "Sensitive Groups"),
    (150, 200, "#ff0000", "Unhealthy"),
    (200, 300, "#8f3f97", "Very Unhealthy"),
    (300, 700, "#7e0023", "Hazardous"),
]

# Max reference values used for progress bar width (% of max)
POLLUTANT_MAX = {
    "pm2_5":                250,
    "pm10":                 430,
    "no2":                  100,
    "o3":                   100,
    "temperature_2m":        50,
    "relative_humidity_2m": 100,
    "wind_speed_10m":        60,
}


def _aqi_color(label: str) -> str:
    return AQI_COLORS.get(label, "#cccccc")

def _aqi_bg(label: str) -> str:
    return AQI_BG.get(label, "#f0f0f0")

def _trend(prev: float, curr: float) -> str:
    if curr > prev + 3:  return "↑"
    if curr < prev - 3:  return "↓"
    return "→"


# ── CSS injection ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Font size +2px  (Streamlit base ≈16px → 18px) ─────────────────── */
    html, body { font-size: 18px !important; }
    p, li, span, div, td, th, label, button { font-size: 18px !important; }
    h1 { font-size: 2.6rem !important; font-weight: 900 !important; }
    h2 { font-size: 2.1rem !important; font-weight: 800 !important; }
    h3 { font-size: 1.75rem !important; font-weight: 700 !important; }
    small, .caption, [data-testid="stCaptionContainer"] * {
        font-size: 15px !important;
    }

    /* ── Keyframes ───────────────────────────────────────────────────────── */
    @keyframes fadeInUp {
        from { opacity:0; transform:translateY(26px); }
        to   { opacity:1; transform:translateY(0);    }
    }
    @keyframes fadeInDown {
        from { opacity:0; transform:translateY(-18px); }
        to   { opacity:1; transform:translateY(0);     }
    }
    @keyframes float {
        0%,100% { transform:translateY(0px);  }
        50%      { transform:translateY(-6px); }
    }
    @keyframes pulseBorder {
        0%,100% { box-shadow: 0 0  0   0px rgba(255,126,0,0.55); }
        50%      { box-shadow: 0 0 22px  8px rgba(255,126,0,0.00); }
    }
    @keyframes blink {
        0%,100% { opacity:1;   }
        50%      { opacity:0.1; }
    }
    @keyframes shimmer {
        0%   { background-position:-400% center; }
        100% { background-position: 400% center; }
    }
    @keyframes slideIn {
        from { opacity:0; transform:translateX(-28px); }
        to   { opacity:1; transform:translateX(0);     }
    }
    @keyframes numberPop {
        0%   { transform:scale(0.55) rotate(-5deg); opacity:0; }
        70%  { transform:scale(1.10) rotate(1deg);  opacity:1; }
        100% { transform:scale(1.00) rotate(0deg);  opacity:1; }
    }

    /* ── Forecast cards ──────────────────────────────────────────────────── */
    .fc-card {
        border-radius: 20px;
        padding: 22px 14px 20px;
        text-align: center;
        transition: transform .28s cubic-bezier(.34,1.56,.64,1),
                    box-shadow .28s ease;
        animation: fadeInUp .55s ease both;
        position: relative;
        overflow: visible;
        border: 1.5px solid rgba(255,255,255,0.08);
    }
    .fc-card::before {
        content:"";
        position:absolute; inset:0;
        background:linear-gradient(135deg,rgba(255,255,255,0.10),rgba(255,255,255,0.00));
        pointer-events:none;
        border-radius:inherit;
    }
    .fc-card:hover {
        transform:translateY(-10px) scale(1.03);
        box-shadow:0 22px 50px rgba(0,0,0,0.42);
    }
    .fc-card-now {
        animation: fadeInUp .5s ease both, pulseBorder 2.6s ease-in-out infinite;
        border:1.5px solid rgba(255,126,0,0.55) !important;
    }
    .fc-card-now::after {
        content:"● LIVE";
        position:absolute; top:10px; right:13px;
        font-size:11px !important;
        font-weight:900;
        letter-spacing:1.8px;
        color:#00e676;
        animation:blink 1.5s ease-in-out infinite;
    }
    .fc-value {
        font-size:3.1rem !important;
        font-weight:900;
        line-height:1.1;
        animation:numberPop .55s cubic-bezier(.34,1.56,.64,1) both;
        font-variant-numeric:tabular-nums;
        letter-spacing:-1px;
    }
    .fc-trend {
        font-size:1.6rem !important;
        vertical-align:middle;
        opacity:.75;
    }
    .fc-horizon {
        font-size:12px !important;
        font-weight:700;
        letter-spacing:2.2px;
        text-transform:uppercase;
        color:#666;
        margin-bottom:8px;
    }
    .fc-label {
        font-size:13px !important;
        font-weight:700;
        margin-top:8px;
        letter-spacing:.3px;
    }
    .fc-delay-0 { animation-delay:0.00s; }
    .fc-delay-1 { animation-delay:0.10s; }
    .fc-delay-2 { animation-delay:0.20s; }
    .fc-delay-3 { animation-delay:0.30s; }

    /* ── Prevent Streamlit columns from clipping hover-lift ─────────────── */
    [data-testid="stColumn"] { overflow:visible !important; }

    /* ── Section headers ─────────────────────────────────────────────────── */
    .section-header {
        font-size:1.25rem !important;
        font-weight:900;
        letter-spacing:.5px;
        animation:slideIn .4s ease both;
        position:relative;
        display:inline-block;
        padding-bottom:5px;
        margin-bottom:16px;
    }
    .section-header::after {
        content:"";
        position:absolute; bottom:0; left:0;
        width:100%; height:3px;
        background:linear-gradient(90deg,#ff7e00 0%,#ffff00 55%,transparent 100%);
        border-radius:2px;
    }

    /* ── Alert banner ────────────────────────────────────────────────────── */
    .aqi-alert { animation:fadeInDown .45s ease both; border-radius:12px; margin-bottom:18px; }

    /* ── Shimmer divider ─────────────────────────────────────────────────── */
    .shimmer-divider {
        height:2px;
        background:linear-gradient(90deg,transparent 0%,#ff7e00 30%,#ffff00 60%,transparent 100%);
        background-size:400% auto;
        animation:shimmer 3s linear infinite;
        border-radius:2px;
        margin:24px 0;
    }

    /* ── Live badge ──────────────────────────────────────────────────────── */
    .live-badge {
        display:inline-block;
        background:linear-gradient(90deg,#00c853,#00e676);
        color:#000;
        font-size:11px !important;
        font-weight:900;
        letter-spacing:2px;
        padding:3px 13px;
        border-radius:20px;
        margin-left:12px;
        vertical-align:middle;
        box-shadow:0 0 12px rgba(0,230,118,0.45);
        animation:blink 2s ease-in-out infinite;
    }

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .app-footer {
        text-align:center;
        font-size:14px !important;
        color:#555;
        margin-top:30px;
        animation:fadeInUp .7s ease both;
        padding:18px;
        border-top:1px solid rgba(255,255,255,0.06);
    }
    .float-emoji {
        display:inline-block;
        animation:float 3.5s ease-in-out infinite;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background:linear-gradient(170deg,#1a1a2e,#16213e) !important;
    }
    [data-testid="stSidebar"] * { font-size:17px !important; }

    /* ── Expander ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] summary {
        font-size:17px !important;
        font-weight:600;
        letter-spacing:.3px;
    }

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width:5px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:#444; border-radius:3px; }
    </style>
    """, unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_historical_data() -> pd.DataFrame | None:
    try:
        df = read_features()
        if not df.empty:
            return df.sort_values("timestamp")
    except Exception:
        pass
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "backfill.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df.sort_values("timestamp")
    return None


# ── Alert banner ──────────────────────────────────────────────────────────────
def render_alert(forecasts: list):
    worst_aqi = max(f["aqi_us"] for f in forecasts)
    worst_label = aqi_label(worst_aqi)
    if worst_aqi > 150:
        color = _aqi_color(worst_label)
        bg    = _aqi_bg(worst_label)
        st.markdown(
            f"""<div class="aqi-alert"
                 style="background:{bg}; border-left:6px solid {color}; padding:14px 22px;">
                <strong style="font-size:17px;">⚠️ Air Quality Alert — Karachi</strong><br>
                Forecast peak: <strong>{worst_aqi:.0f} US AQI</strong>
                — <span style="color:{color}; font-weight:800;">{worst_label}</span>.<br>
                <span style="font-size:15px !important; color:#666;">
                Sensitive individuals should avoid prolonged outdoor exposure.</span>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Forecast cards ────────────────────────────────────────────────────────────
def render_forecast_cards(forecasts: list, current_aqi: float | None):
    cols = st.columns(4)

    if current_aqi is not None:
        label = aqi_label(current_aqi)
        bg    = _aqi_bg(label)
        color = _aqi_color(label)
        cols[0].markdown(
            f"""<div class="fc-card fc-card-now fc-delay-0" style="background:{bg};">
                <div class="fc-horizon">NOW</div>
                <div class="fc-value" style="color:{color};">{current_aqi:.0f}</div>
                <div class="fc-label" style="color:{color};">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    horizon_labels = {24: "+1 Day", 48: "+2 Days", 72: "+3 Days"}
    delays         = ["fc-delay-1", "fc-delay-2", "fc-delay-3"]
    prev_val       = current_aqi if current_aqi is not None else (forecasts[0]["aqi_us"] if forecasts else 0)

    for col, fc, delay in zip(cols[1:], forecasts, delays):
        label  = fc["label"]
        bg     = _aqi_bg(label)
        color  = _aqi_color(label)
        arrow  = _trend(prev_val, fc["aqi_us"])
        hlabel = horizon_labels.get(fc["horizon_h"], f"+{fc['horizon_h']//24}d")
        col.markdown(
            f"""<div class="fc-card {delay}" style="background:{bg};">
                <div class="fc-horizon">{hlabel}</div>
                <div class="fc-value" style="color:{color};">
                    {fc['aqi_us']:.0f}
                    <span class="fc-trend">{arrow}</span>
                </div>
                <div class="fc-label" style="color:{color};">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        prev_val = fc["aqi_us"]


# ── History + forecast chart ──────────────────────────────────────────────────
def render_history_chart(hist_df: pd.DataFrame, forecasts: list, last_ts):
    cutoff = (
        pd.Timestamp(last_ts) - timedelta(days=7)
        if last_ts else hist_df["timestamp"].max() - timedelta(days=7)
    )
    recent = hist_df[hist_df["timestamp"] >= cutoff].copy()

    fig = go.Figure()

    # Coloured AQI zone bands — makes it immediately obvious which zone the line is in
    for y0, y1, band_color, _ in AQI_BANDS:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=band_color, opacity=0.07, line_width=0)

    # Historical line
    fig.add_trace(go.Scatter(
        x=recent["timestamp"],
        y=recent["aqi_us"],
        mode="lines",
        name="Historical AQI",
        line=dict(color="#4fc3f7", width=2.5),
    ))

    # "Now" dot on the most recent historical point
    if not recent.empty:
        last_val = recent["aqi_us"].dropna()
        if not last_val.empty:
            lv = last_val.iloc[-1]
            lt = recent.loc[last_val.index[-1], "timestamp"]
            fig.add_trace(go.Scatter(
                x=[lt], y=[lv],
                mode="markers",
                name="Current",
                marker=dict(size=13, color="#4fc3f7",
                            line=dict(color="#fff", width=2)),
                showlegend=False,
            ))
            fig.add_annotation(
                x=lt, y=lv,
                text=f" Now: {lv:.0f}",
                showarrow=False,
                font=dict(size=13, color="#4fc3f7"),
                xanchor="left",
            )

    # Forecast points
    if last_ts:
        base_ts = pd.Timestamp(last_ts)
        fc_times = [base_ts + timedelta(hours=f["horizon_h"]) for f in forecasts]
        fc_vals  = [f["aqi_us"] for f in forecasts]
        fig.add_trace(go.Scatter(
            x=fc_times, y=fc_vals,
            mode="markers+lines",
            name="Forecast",
            line=dict(color="#ff7e00", width=2.5, dash="dash"),
            marker=dict(size=13, symbol="diamond", color="#ff7e00",
                        line=dict(color="#fff", width=2)),
        ))

    # Threshold dotted lines with colour labels
    for threshold, label, color in [
        (50,  "Good",      "#00e400"),
        (100, "Moderate",  "#ffff00"),
        (150, "Sensitive", "#ff7e00"),
        (200, "Unhealthy", "#ff0000"),
    ]:
        fig.add_hline(
            y=threshold,
            line_dash="dot", line_color=color, line_width=1.5,
            annotation_text=label,
            annotation_position="bottom right",
            annotation_font_size=13, annotation_font_color=color,
        )

    fig.update_layout(
        title=dict(text="Historical AQI (7 days) + 3-Day Forecast", font=dict(size=19)),
        xaxis_title="Time",
        yaxis_title="US AQI",
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=16)),
        height=450,
        margin=dict(l=0, r=0, t=65, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=14)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=14)),
        font=dict(size=16),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Pollutant snapshot (native Streamlit — avoids markdown code-block escaping) ─
def render_pollutant_snapshot(raw_row: dict):
    POLLUTANTS = [
        ("pm2_5",               "PM2.5 (µg/m³)",      "🌫️"),
        ("pm10",                "PM10 (µg/m³)",       "💨"),
        ("no2",                 "NO₂ (µg/m³)",        "🏭"),
        ("o3",                  "O₃ (µg/m³)",         "☀️"),
        ("temperature_2m",      "Temperature (°C)", "🌡️"),
        ("relative_humidity_2m","Humidity (%)",       "💧"),
        ("wind_speed_10m",      "Wind Speed (km/h)",  "🌬️"),
    ]

    for key, label, icon in POLLUTANTS:
        raw = raw_row.get(key)
        try:
            val = float(raw)
            vstr = f"{val:.1f}"
            ratio = min(1.0, val / POLLUTANT_MAX.get(key, 100))
        except (TypeError, ValueError):
            vstr = "N/A"
            ratio = 0.0

        col_label, col_val, col_bar = st.columns([2.8, 1.2, 4.5], gap="small")
        with col_label:
            st.markdown(f"{icon} **{label}**")
        with col_val:
            st.markdown(f"**{vstr}**")
        with col_bar:
            st.progress(ratio)


# ── SHAP panel ────────────────────────────────────────────────────────────────
def render_shap(feature_row: dict):
    import joblib

    model_path = os.path.join(os.path.dirname(__file__), "..", "models_artifacts", "best_model.pkl")
    if not os.path.exists(model_path):
        st.info("SHAP not available — run the training pipeline first.")
        return

    try:
        import shap
        pipeline     = joblib.load(model_path)
        feature_cols = load_training_feature_columns()
        X            = pd.DataFrame([feature_row])[feature_cols].values

        estimator = pipeline.named_steps.get("model", pipeline)
        inner     = estimator.estimators_[0] if hasattr(estimator, "estimators_") else estimator

        explainer = (
            shap.TreeExplainer(inner)
            if hasattr(inner, "feature_importances_")
            else shap.LinearExplainer(inner, X)
        )

        X_scaled = pipeline.named_steps["scaler"].transform(X) if "scaler" in pipeline.named_steps else X
        shap_vals = explainer(X_scaled)

        top_n = 8
        importances = pd.DataFrame({
            "Feature":     feature_cols,
            "SHAP Value":  np.abs(shap_vals.values[0]),
        }).sort_values("SHAP Value", ascending=False).head(top_n)

        st.subheader("Top Feature Contributions (latest prediction, +24h horizon)")

        max_shap = importances["SHAP Value"].max()
        colors   = [
            f"rgba(255,126,0,{0.35 + 0.65 * v / max_shap:.2f})"
            for v in importances["SHAP Value"]
        ]

        fig = go.Figure(go.Bar(
            x=importances["SHAP Value"],
            y=importances["Feature"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}" for v in importances["SHAP Value"]],
            textposition="outside",
            textfont=dict(size=13),
        ))
        fig.update_layout(
            xaxis_title="|SHAP value|",
            yaxis_title="Feature",
            height=340,
            margin=dict(l=0, r=60, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=14)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=14)),
            font=dict(size=16),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"SHAP computation skipped: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # Header
    st.markdown(
        '<div style="animation:fadeInDown .5s ease both; margin-bottom:4px;">'
        '<span style="font-size:2.3rem; font-weight:900;">Karachi AQI Forecaster</span>'
        '<span class="live-badge">LIVE</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Real-time 3-day Air Quality Index prediction · Open-Meteo + MongoDB + scikit-learn")

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    use_local  = st.sidebar.toggle("Use local model (no MongoDB)", value=False)
    show_shap  = st.sidebar.toggle("Show SHAP explanation", value=True)
    refresh    = st.sidebar.button("🔄 Refresh prediction", use_container_width=True)

    # Prediction
    if "prediction" not in st.session_state or refresh:
        with st.spinner("Running inference..."):
            try:
                st.session_state["prediction"] = predict(local=use_local)
                st.session_state["pred_error"]  = None
            except Exception as exc:
                st.session_state["pred_error"] = str(exc)

    if st.session_state.get("pred_error"):
        st.error(f"Prediction failed: {st.session_state['pred_error']}")
        st.info("Make sure MongoDB credentials are set and the training pipeline has registered a model.")
        return

    result      = st.session_state["prediction"]
    forecasts   = result["forecasts"]
    current_aqi = result.get("latest_actual")
    last_ts     = result.get("latest_timestamp")
    feature_row = result.get("feature_row", {})

    # Load historical data — also the source of raw pollutant concentrations
    hist_df = load_historical_data()

    # Build raw_row: prefer the latest MongoDB/CSV row (has pm2_5, pm10, humidity, etc.
    # that were excluded from feature_row by the correlation pruning step).
    raw_row: dict = {}
    if hist_df is not None and not hist_df.empty:
        raw_row = hist_df.sort_values("timestamp").iloc[-1].to_dict()
    for k, v in feature_row.items():
        if k not in raw_row or raw_row.get(k) is None:
            raw_row[k] = v

    # Alert
    render_alert(forecasts)

    # Forecast cards
    st.markdown('<div class="section-header">Current Conditions &amp; 3-Day Forecast</div>',
                unsafe_allow_html=True)
    render_forecast_cards(forecasts, current_aqi)

    st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

    # Chart
    if hist_df is not None and "aqi_us" in hist_df.columns:
        render_history_chart(hist_df, forecasts, last_ts)
    else:
        st.info("No historical data. Run `python src/pipelines/backfill.py --days 90` to populate MongoDB.")

    st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

    # Pollutant snapshot
    if raw_row:
        st.markdown('<div class="section-header">Current Pollutant Snapshot</div>',
                    unsafe_allow_html=True)
        render_pollutant_snapshot(raw_row)

    st.markdown('<div class="shimmer-divider"></div>', unsafe_allow_html=True)

    # SHAP
    if show_shap and feature_row:
        with st.expander("🔍 Feature Importance (SHAP)", expanded=False):
            render_shap(feature_row)

    # Footer
    generated_at = result.get("generated_at", "")
    st.markdown(
        f'<div class="app-footer">'
        f'Prediction generated at <strong>{generated_at} UTC</strong>'
        f' &nbsp;·&nbsp; Karachi Air Quality Forecaster'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
