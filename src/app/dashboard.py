"""Streamlit Interactive Production Dashboard for Multimodal Risk Copilot."""
from __future__ import annotations
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.risk.metrics import risk_metrics
from src.risk.monte_carlo import simulate_monte_carlo
from src.time_series.preprocessor import prepare_prices
from src.time_series.forecaster import forecast_returns, forecast_volatility
from src.vision.chart_renderer import render_chart
from src.vision.pattern_agent import analyze_chart
from src.llm_engine.risk_agent import RiskCopilotAgent

st.set_page_config(
    page_title="Multimodal Risk Copilot",
    page_icon="📈",
    layout="wide",
)

st.title("🛡️ Multimodal Risk Copilot")
st.markdown("### Institutional Quantitative Risk, Time-Series & Vision Intelligence Platform")

st.sidebar.header("Configuration & Data Source")
sample_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_prices.csv")
uploaded_file = st.sidebar.file_uploader("Upload Price History (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(sample_path):
    st.sidebar.info("Using included sample price dataset.")
    df = pd.read_csv(sample_path)
else:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=250)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 250)))
    df = pd.DataFrame({"Date": dates, "Close": prices, "High": prices*1.01, "Low": prices*0.99, "Open": prices, "Volume": 1000000})

col_name = st.sidebar.selectbox("Price Column", options=[c for c in df.columns if c != "Date"], index=0)
model_type = st.sidebar.selectbox("Monte Carlo Model", ["gbm", "jump_diffusion"])
mc_days = st.sidebar.slider("Horizon (Days)", min_value=30, max_value=504, value=252, step=30)
mc_paths = st.sidebar.slider("Simulation Paths", min_value=500, max_value=10000, value=2500, step=500)

df_prep = prepare_prices(df, price_col=col_name)
returns = df_prep["log_return"].dropna().to_numpy()

col1, col2, col3, col4 = st.columns(4)
metrics = risk_metrics(returns)
col1.metric("Historical VaR (95%)", f"{metrics['var_95']:.2%}")
col2.metric("Historical CVaR (95%)", f"{metrics['cvar_95']:.2%}")
col3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market & Technical Vision", "🎲 Monte Carlo Simulation", "⏳ Volatility & Trends", "📝 CRO Executive Report"])

with tab1:
    st.subheader("Technical Multimodal Pattern Scan")
    pattern_info = analyze_chart("", price_df=df)
    st.info(f"Detected Regime: **{pattern_info.get('regime')}** | Support: **{pattern_info.get('key_support', 0):.2f}** | Resistance: **{pattern_info.get('key_resistance', 0):.2f}**")
    
    chart_output = "/tmp/dashboard_chart.png"
    render_chart(df, chart_output, title=f"Technical Chart - {col_name}")
    if os.path.exists(chart_output):
        st.image(chart_output, use_column_width=True)

with tab2:
    st.subheader("Forward Monte Carlo Distribution")
    curr_price = float(df_prep[col_name].iloc[-1])
    mu = float(returns.mean() * 252)
    sigma = float(returns.std(ddof=1) * np.sqrt(252))
    mc = simulate_monte_carlo(curr_price, mu=mu, sigma=sigma, days=mc_days, n_paths=mc_paths, model=model_type)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(mc["terminal_prices"], bins=50, alpha=0.75, color="#1f77b4", edgecolor="black")
    ax.axvline(curr_price, color="red", linestyle="--", label=f"Spot: {curr_price:.2f}")
    ax.set_title("Terminal Price Scenario Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
    
    st.write(pd.DataFrame([mc["summary"]]))

with tab3:
    st.subheader("Time Series Forecasts")
    vol_forecast = forecast_volatility(returns, horizon=20)
    ret_forecast = forecast_returns(returns, horizon=20)
    st.write(f"Annualized Projected Volatility: **{vol_forecast['annualized_volatility']:.2%}**")
    st.write(f"Cumulative Return Outlook (20d): **{ret_forecast['cumulative_return']:.2%}**")

with tab4:
    st.subheader("Executive AI Risk Memo")
    agent = RiskCopilotAgent()
    memo = agent.synthesize_report(
        risk_metrics=metrics,
        mc_summary=mc["summary"],
        ts_forecast=ret_forecast,
        vol_forecast=vol_forecast,
        pattern_data=pattern_info
    )
    st.markdown(memo)
