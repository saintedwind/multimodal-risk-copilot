"""Leakage-conscious price-series transforms."""
from __future__ import annotations
import numpy as np
import pandas as pd

def prepare_prices(data: pd.DataFrame, price_col: str="Close", span: int=20) -> pd.DataFrame:
    """Return prices, log returns, EWMA volatility and trend/residual components."""
    if price_col not in data: raise ValueError(f"missing {price_col!r}")
    out=data.copy(); price=pd.to_numeric(out[price_col],errors="coerce")
    if (price.dropna()<=0).any(): raise ValueError("prices must be positive")
    out["log_return"]=np.log(price).diff()
    out["ewma_volatility"]=out["log_return"].ewm(span=span,adjust=False,min_periods=2).std(bias=False)*np.sqrt(252)
    trend=price.ewm(span=max(2,span*3),adjust=False).mean()
    out["trend"]=trend; out["residual"]=price-trend
    return out

def decompose_trend(series: pd.Series, span: int=60) -> pd.DataFrame:
    """Deterministic exponential trend decomposition."""
    s=pd.to_numeric(series,errors="coerce"); trend=s.ewm(span=span,adjust=False,min_periods=1).mean()
    return pd.DataFrame({"observed":s,"trend":trend,"residual":s-trend},index=s.index)
