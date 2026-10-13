"""Strict, chronological, single-instrument daily price transforms."""

import numpy as np
import pandas as pd


def prepare_prices(data, price_col="Close", span=20):
    if price_col not in data:
        raise ValueError(f"Missing price column: {price_col}")
    if not 6 <= len(data) <= 20000:
        raise ValueError("Require 6..20000 price rows")
    if not isinstance(span, int) or span < 2:
        raise ValueError("Span must be at least 2")
    out = data.copy()
    if "Symbol" in out and out["Symbol"].nunique(dropna=False) != 1:
        raise ValueError("One instrument per analysis is required")
    if "Date" in out:
        dates = pd.to_datetime(out["Date"], errors="raise")
        if (
            dates.isna().any()
            or dates.dt.normalize().duplicated().any()
            or not dates.is_monotonic_increasing
        ):
            raise ValueError(
                "Dates must be ascending with at most one observation per calendar day"
            )
        out["Date"] = dates
    price = pd.to_numeric(out[price_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Prices must be finite and positive")
    out[price_col] = price
    out["simple_return"] = pd.Series(price, index=out.index).pct_change(
        fill_method=None
    )
    out["log_return"] = (
        np.log(price).diff()
        if hasattr(price, "diff")
        else pd.Series(np.log(price), index=out.index).diff()
    )
    out["ewma_volatility"] = out["log_return"].ewm(
        span=span, adjust=False, min_periods=2
    ).std(bias=False) * np.sqrt(252)
    out["trend"] = out[price_col].ewm(span=span * 3, adjust=False).mean()
    out["residual"] = out[price_col] - out["trend"]
    if all(c in out for c in ["Open", "High", "Low", "Close"]):
        vals = out[["Open", "High", "Low", "Close"]].apply(
            pd.to_numeric, errors="raise"
        )
        if (
            not np.isfinite(vals.to_numpy()).all()
            or (vals <= 0).any().any()
            or (vals.High < vals[["Open", "Close"]].max(axis=1)).any()
            or (vals.Low > vals[["Open", "Close"]].min(axis=1)).any()
        ):
            raise ValueError("Inconsistent OHLC bars")
        out[["Open", "High", "Low", "Close"]] = vals
    return out


def decompose_trend(series, span=60):
    s = pd.to_numeric(series, errors="raise")
    trend = s.ewm(span=span, adjust=False).mean()
    return pd.DataFrame({"observed": s, "trend": trend, "residual": s - trend})
