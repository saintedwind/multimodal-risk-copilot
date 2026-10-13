"""FastAPI REST Service for Multimodal Risk Copilot."""
from __future__ import annotations
import os
import io
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from src.risk.metrics import risk_metrics
from src.risk.monte_carlo import simulate_monte_carlo
from src.time_series.forecaster import forecast_returns, forecast_volatility
from src.time_series.preprocessor import prepare_prices
from src.vision.pattern_agent import analyze_chart
from src.vision.chart_renderer import render_chart
from src.llm_engine.risk_agent import RiskCopilotAgent

app = FastAPI(
    title="Multimodal Risk Copilot API",
    description="Institutional-grade Quantitative Risk & Multimodal Vision Copilot",
    version="1.0.0"
)

class RiskRequest(BaseModel):
    returns: List[float] = Field(..., description="Array of periodic returns")
    risk_free_rate: float = Field(0.0, description="Annualized risk-free rate")
    horizon_days: int = Field(252, description="Monte Carlo horizon in days")
    n_paths: int = Field(5000, description="Monte Carlo scenario paths")
    model: str = Field("gbm", description="Monte Carlo model: gbm or jump_diffusion")
    initial_price: float = Field(100.0, description="Underlying base price")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Multimodal Risk Copilot API"}

@app.post("/analyze/metrics")
def calculate_metrics(req: RiskRequest):
    try:
        metrics = risk_metrics(req.returns, risk_free_rate=req.risk_free_rate)
        arr = np.array(req.returns)
        mu = float(arr.mean() * 252)
        sigma = float(arr.std(ddof=1) * np.sqrt(252))
        mc = simulate_monte_carlo(
            initial_price=req.initial_price,
            mu=mu,
            sigma=sigma,
            days=req.horizon_days,
            n_paths=req.n_paths,
            model=req.model,
        )
        ts_ret = forecast_returns(req.returns, horizon=min(30, req.horizon_days))
        ts_vol = forecast_volatility(req.returns, horizon=min(30, req.horizon_days))
        
        return {
            "risk_metrics": metrics,
            "monte_carlo_summary": mc["summary"],
            "return_forecast": ts_ret,
            "volatility_forecast": ts_vol
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze/full")
async def analyze_full_dataset(
    file: UploadFile = File(...),
    price_col: str = Form("Close"),
    initial_price: Optional[float] = Form(None)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if price_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column {price_col} not found in CSV")
        
        df_prep = prepare_prices(df, price_col=price_col)
        returns = df_prep["log_return"].dropna().tolist()
        
        metrics = risk_metrics(returns)
        start_price = float(initial_price if initial_price is not None else df_prep[price_col].iloc[-1])
        mu = float(np.mean(returns) * 252)
        sigma = float(np.std(returns, ddof=1) * np.sqrt(252))
        
        mc = simulate_monte_carlo(start_price, mu=mu, sigma=sigma, days=252, n_paths=2000)
        ts_ret = forecast_returns(returns, horizon=20)
        ts_vol = forecast_volatility(returns, horizon=20)
        patterns = analyze_chart("", price_df=df)
        
        agent = RiskCopilotAgent()
        report = agent.synthesize_report(
            risk_metrics=metrics,
            mc_summary=mc["summary"],
            ts_forecast=ts_ret,
            vol_forecast=ts_vol,
            pattern_data=patterns
        )
        
        return {
            "metrics": metrics,
            "patterns": patterns,
            "monte_carlo": mc["summary"],
            "executive_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
