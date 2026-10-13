"""Local review dashboard; shared calculations and explicit provenance."""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.service import analyze_prices
from src.vision.chart_renderer import render_chart

st.set_page_config(page_title="Market Risk Review Desk", layout="wide")
st.markdown(
    "<style>.stApp{background:#f7f4ec}.stApp h1,.stApp h2,.stApp h3{font-family:Georgia,serif!important}div[data-testid='stMetric']{border-top:2px solid #315748;padding:12px}</style>",
    unsafe_allow_html=True,
)
st.title("Market Risk Review Desk")
st.caption("DAILY OBSERVATIONS / REPRODUCIBLE SCENARIOS / SOURCE REVIEW")
st.info(
    "Included price data has unverified provenance. This demo does not represent a validated investment or visual prediction system."
)
uploaded = st.sidebar.file_uploader("Price history CSV", type="csv", max_upload_size=2)
try:
    if uploaded and uploaded.size > 2_000_000:
        raise ValueError("CSV exceeds 2 MB")
    df = pd.read_csv(
        uploaded
        if uploaded
        else Path(__file__).resolve().parents[2] / "data/sample_prices.csv",
        nrows=20001,
    )
    if len(df) > 20000:
        raise ValueError("CSV exceeds 20000 rows")
    if "Symbol" in df:
        symbol = st.sidebar.selectbox("Instrument", sorted(df.Symbol.dropna().unique()))
        df = df[df.Symbol == symbol].reset_index(drop=True)
    col = st.sidebar.selectbox(
        "Price column",
        [c for c in df if c not in ["Date", "Symbol"]],
        index=list(c for c in df if c not in ["Date", "Symbol"]).index("Close")
        if "Close" in df
        else 0,
    )
    days = st.sidebar.slider("Trading-day horizon", 1, 252, 60)
    paths = st.sidebar.select_slider(
        "Scenario paths", options=[500, 1000, 2000, 5000], value=1000
    )
    seed = st.sidebar.number_input(
        "Random seed", min_value=0, max_value=2**32 - 1, value=42
    )
    model = st.sidebar.selectbox("Scenario model", ["gbm", "jump_diffusion"])
    jump = {}
    if model == "jump_diffusion":
        jump = {
            "jump_intensity": st.sidebar.number_input(
                "Annual jump intensity", 0.0, 50.0, 2.0
            ),
            "jump_mean": st.sidebar.number_input("Mean log jump", -1.0, 1.0, -0.05),
            "jump_sigma": st.sidebar.number_input("Log jump deviation", 0.0, 1.0, 0.1),
        }
    r = analyze_prices(
        df, col, horizon_days=days, n_paths=paths, seed=int(seed), model=model, **jump
    )
    a, b, c, d = st.columns(4)
    m = r["risk_metrics"]
    a.metric("Daily loss VaR / 95%", f"{m['var_95']:.2%}")
    b.metric("Historical drawdown", f"{m['max_drawdown']:.2%}")
    c.metric(
        "Scenario loss probability",
        f"{r['monte_carlo_summary']['loss_probability']:.1%}",
    )
    d.metric("Trading observations", m["observations"])
    overview, scenario, memo, details = st.tabs(
        ["Price evidence", "Scenario assumptions", "Review notes", "Download & audit"]
    )
    with overview:
        st.line_chart(df.set_index("Date")[col] if "Date" in df else df[col])
        if all(x in df for x in ["Open", "High", "Low", "Close"]) and col == "Close":
            image = io.BytesIO()
            render_chart(df, image, title="OHLC evidence / last 120 observations")
            st.image(image.getvalue())
            st.json(r["patterns"])
    with scenario:
        st.json(r["monte_carlo_summary"])
        st.write(
            "GBM fitted to log returns; jump parameters are scenario assumptions. Same input and seed reproduce the scenario."
        )
        st.json(r["volatility_forecast"])
    with memo:
        st.markdown(r["executive_report"])
        st.caption("Report mode: " + r["report_status"])
    with details:
        import json

        st.json(r["assumptions"])
        st.download_button(
            "Download analysis JSON",
            json.dumps(r, indent=2),
            file_name="risk-review.json",
            mime="application/json",
        )
except (ValueError, KeyError, IndexError) as e:
    st.error(str(e))
    st.stop()
