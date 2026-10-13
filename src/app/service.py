"""Shared analysis contract used by both UI and HTTP API."""

import numpy as np

from src.llm_engine.risk_agent import RiskCopilotAgent
from src.risk.metrics import _arr, risk_metrics
from src.risk.monte_carlo import simulate_monte_carlo
from src.time_series.forecaster import forecast_returns, forecast_volatility
from src.time_series.preprocessor import prepare_prices
from src.vision.pattern_agent import analyze_chart


def analyze_returns(
    returns,
    risk_free_rate=0.0,
    horizon_days=60,
    n_paths=1000,
    model="gbm",
    initial_price=100.0,
    seed=42,
    jump_intensity=0.0,
    jump_mean=0.0,
    jump_sigma=0.0,
):
    x = _arr(returns)
    if len(x) < 5:
        raise ValueError("At least five daily returns required")
    log = np.log1p(x)
    sigma = float(log.std(ddof=1) * np.sqrt(252))
    mu = float(log.mean() * 252 + 0.5 * sigma * sigma)
    mc = simulate_monte_carlo(
        initial_price,
        mu,
        sigma,
        days=horizon_days,
        n_paths=n_paths,
        model=model,
        seed=seed,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_sigma=jump_sigma,
    )
    return {
        "risk_metrics": risk_metrics(x, risk_free_rate),
        "monte_carlo_summary": mc["summary"],
        "return_forecast": forecast_returns(x, min(30, horizon_days)),
        "volatility_forecast": forecast_volatility(x, min(30, horizon_days)),
        "assumptions": {
            "data_frequency": "daily trading observations",
            "return_type": "simple",
            "annualization": 252,
            "drift_fit": "mean(log1p(r))*252 + sigma^2/2",
            "provenance": "provided data, not independently verified",
        },
        "warnings": [
            "Historical tail estimates are unstable with limited samples.",
            "Jump parameters are user assumptions, not calibrated estimates.",
        ],
    }


def analyze_prices(df, price_col="Close", **options):
    frame = prepare_prices(df, price_col)
    instrument = str(frame.Symbol.iloc[0]) if "Symbol" in frame else "unspecified"
    out = analyze_returns(
        frame.simple_return.dropna().tolist(),
        initial_price=float(frame[price_col].iloc[-1]),
        **options,
    )
    out["patterns"] = (
        analyze_chart("", frame)
        if all(c in frame for c in ["Open", "High", "Low", "Close"])
        and price_col == "Close"
        else {
            "regime": None,
            "patterns_detected": [],
            "multimodal_source": "ohlc_not_available",
        }
    )
    out["assumptions"]["instrument"] = instrument
    a = RiskCopilotAgent()
    out["executive_report"] = a.synthesize_report(
        out["risk_metrics"],
        out["monte_carlo_summary"],
        out["return_forecast"],
        out["volatility_forecast"],
        out["patterns"],
    )
    out["report_status"] = a.last_status
    if hasattr(a, "last_usage"):
        out["local_model_usage"] = a.last_usage
    return out
