"""Daily simple-return statistics; undefined ratios are null, never enormous."""

from statistics import NormalDist

import numpy as np


def _arr(returns):
    x = np.asarray(returns, dtype=float)
    if (
        x.ndim != 1
        or not 2 <= len(x) <= 20000
        or not np.isfinite(x).all()
        or (x <= -1).any()
    ):
        raise ValueError("Require 2..20000 finite daily simple returns greater than -1")
    return x


def risk_metrics(returns, risk_free_rate=0.0):
    x = _arr(returns)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("Invalid annual risk-free rate")
    losses = -x
    out = {}
    vol = float(x.std(ddof=1))
    rf = np.expm1(np.log1p(risk_free_rate) / 252)
    for level in (0.95, 0.99):
        q = float(np.quantile(losses, level, method="linear"))
        tail = losses[losses >= q]
        key = int(level * 100)
        out[f"var_{key}"] = q
        out[f"cvar_{key}"] = float(tail.mean())
        out[f"historical_var_{key}"] = q
        out[f"historical_cvar_{key}"] = float(tail.mean())
        out[f"parametric_var_{key}"] = float(
            -x.mean() + vol * NormalDist().inv_cdf(level)
        )
    logwealth = np.cumsum(np.log1p(x))
    logpeak = np.maximum.accumulate(np.r_[0.0, logwealth])[1:]
    dd = np.expm1(logwealth - logpeak)
    annual = float(np.expm1(logwealth[-1] * 252 / len(x)))
    downside = float(np.sqrt(np.mean(np.minimum(x - rf, 0) ** 2)))
    ratio = lambda a, b: float(a / b) if b > 1e-12 else None
    out.update(
        max_drawdown=float(dd.min()),
        sharpe=ratio(float((x.mean() - rf) * np.sqrt(252)), vol),
        sortino=ratio(float((x.mean() - rf) * np.sqrt(252)), downside),
        calmar=ratio(annual, abs(float(dd.min()))),
        annualized_return=annual,
        observations=len(x),
        return_type="daily_simple",
        periods_per_year=252,
        tail_definition="sample conditional mean above linear loss quantile",
    )
    if not all(
        v is None or not isinstance(v, float) or np.isfinite(v) for v in out.values()
    ):
        raise ValueError("Numerical range exceeded")
    return out
