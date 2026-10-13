"""AR(1) baseline and EWMA on daily simple returns, with truthful method labels."""

import numpy as np

from src.risk.metrics import _arr


def forecast_returns(returns, horizon=10, method="auto"):
    x = _arr(returns)
    if len(x) < 5 or not isinstance(horizon, int) or not 1 <= horizon <= 504:
        raise ValueError("Require five returns and horizon 1..504")
    if method not in {"auto", "ar", "holt_winters"}:
        raise ValueError("Unsupported method")
    used = "ar1"
    fallback = None
    if method == "holt_winters":
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            if len(x) < 12:
                raise ValueError("At least 12 observations required")
            pred = np.asarray(
                ExponentialSmoothing(x, trend="add", initialization_method="estimated")
                .fit()
                .forecast(horizon),
                float,
            )
            used = "holt_winters"
        except (ImportError, ValueError, RuntimeError):
            fallback = "holt_winters_unavailable"
    if used == "ar1":
        mean = float(x.mean())
        centered = x - mean
        den = float(np.dot(centered[:-1], centered[:-1]))
        phi = (
            float(np.clip(np.dot(centered[1:], centered[:-1]) / den, -0.95, 0.95))
            if den > 1e-12
            else 0.0
        )
        pred = np.array(
            [mean + phi ** (i + 1) * (x[-1] - mean) for i in range(horizon)]
        )
    if not np.isfinite(pred).all() or (pred <= -1).any():
        raise ValueError("Forecast outside numerical domain")
    return {
        "method": used,
        "fallback_reason": fallback,
        "mean_daily_return": pred.tolist(),
        "cumulative_return": float(np.expm1(np.log1p(pred).sum())),
        "return_type": "daily_simple",
        "horizon_days": horizon,
    }


def forecast_volatility(returns, horizon=10, decay=0.94):
    x = _arr(returns)
    if (
        len(x) < 3
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 504
        or not 0 < decay < 1
    ):
        raise ValueError("Invalid EWMA inputs")
    v = float(np.var(x, ddof=1))
    for r in x:
        v = decay * v + (1 - decay) * float(r * r)
    return {
        "daily_volatility": [float(np.sqrt(v))] * horizon,
        "annualized_volatility": float(np.sqrt(v * 252)),
        "method": "ewma_constant_forward_variance",
        "horizon_days": horizon,
    }
