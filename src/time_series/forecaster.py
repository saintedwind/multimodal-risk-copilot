"""Compact forecasting utilities with optional statistical-model fallbacks."""
from __future__ import annotations
import numpy as np
import pandas as pd

def forecast_returns(returns, horizon=10, method="auto"):
    """Forecast daily returns; uses AR(1), Holt-Winters when available, otherwise robust AR(1)."""
    x=pd.Series(returns,dtype=float).dropna().to_numpy()
    if len(x)<5 or horizon<1: raise ValueError("at least five returns and positive horizon required")
    if method not in {"auto","ar","holt_winters"}: raise ValueError("unsupported method")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        if method=="holt_winters" and len(x)>=12:
            fit=ExponentialSmoothing(x,trend="add",initialization_method="estimated").fit(optimized=True)
            pred=np.asarray(fit.forecast(horizon),float)
        else: raise ImportError
    except Exception:
        mean=float(np.mean(x)); centered=x-mean
        phi=float(np.dot(centered[1:],centered[:-1])/(np.dot(centered[:-1],centered[:-1])+1e-12))
        phi=float(np.clip(phi,-.95,.95)); pred=np.array([mean+phi**(i+1)*(x[-1]-mean) for i in range(horizon)])
    return {"method":"holt_winters" if method=="holt_winters" and len(x)>=12 else "ar1_fallback","mean_daily_return":pred.tolist(),"cumulative_return":float(np.prod(1+pred)-1)}

def forecast_volatility(returns, horizon=10, decay=.94):
    """GARCH-like EWMA variance forecast with a stable unconditional fallback."""
    x=np.asarray(pd.Series(returns,dtype=float).dropna());
    if len(x)<3 or horizon<1: raise ValueError("insufficient returns or invalid horizon")
    var=float(np.var(x,ddof=1)); v=float(np.var(x[:min(30,len(x))],ddof=1))
    for r in x: v=decay*v+(1-decay)*float(r*r)
    values=[v]
    for _ in range(1,horizon): values.append(decay*values[-1]+(1-decay)*var)
    return {"daily_volatility":np.sqrt(values).tolist(),"annualized_volatility":float(np.sqrt(np.mean(values)*252)),"method":"ewma_garch_like"}
