"""One-step risk evaluation using earlier simple returns and zero conditional mean."""

from statistics import NormalDist
import numpy as np
from scipy.special import xlogy
from scipy.stats import binomtest, chi2
from src.risk.metrics import _arr


def coverage_diagnostics(hits, alpha):
    h = np.asarray(hits)
    if h.ndim != 1 or len(h) < 2 or not np.isin(h, [0, 1]).all() or not 0 < alpha < 1:
        raise ValueError("Invalid coverage inputs")
    h = h.astype(int)
    n, k = len(h), int(h.sum())
    transitions = [
        int(np.sum((h[:-1] == a) & (h[1:] == b)))
        for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]
    ]
    n00, n01, n10, n11 = transitions
    independence = None
    if n00 + n01 and n10 + n11:
        p = (n01 + n11) / (n - 1)
        p0, p1 = n01 / (n00 + n01), n11 / (n10 + n11)
        ll_null = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
        ll_alt = (
            xlogy(n00, 1 - p0) + xlogy(n01, p0) + xlogy(n10, 1 - p1) + xlogy(n11, p1)
        )
        lr = max(0.0, float(2 * (ll_alt - ll_null)))
        independence = {
            "lr": lr,
            "asymptotic_p_value": float(chi2.sf(lr, 1)),
            "caution": "Approximation unreliable with sparse transitions",
        }
    return {
        "observations": n,
        "exceptions": k,
        "exception_rate": k / n,
        "expected_exception_rate": 1 - alpha,
        "exact_binomial_p_value": float(binomtest(k, n, 1 - alpha).pvalue),
        "transitions_00_01_10_11": transitions,
        "independence": independence,
    }


def backtest_risk(returns, window=252, decay=0.94, alpha=0.95, start_index=None):
    x = _arr(returns)
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or not 20 <= window <= 1000
        or not 0 < decay < 1
        or not 0.5 < alpha < 1
    ):
        raise ValueError("Invalid backtest parameters")
    start = window if start_index is None else start_index
    if isinstance(start, bool) or not isinstance(start, int) or start < window:
        raise ValueError("Start must follow training window")
    if len(x) - start < 20:
        return {"status": "insufficient_history", "minimum_observations": start + 20}
    variance = float(np.mean(x[:window] ** 2))
    for i in range(window, start):
        variance = decay * variance + (1 - decay) * x[i] ** 2
    paths = {
        name: {"variance": [], "var": []}
        for name in ["rolling_normal", "ewma_normal", "historical"]
    }
    z = NormalDist().inv_cdf(alpha)
    for i in range(start, len(x)):
        train = x[i - window : i]
        rolling = max(float(np.mean(train**2)), 1e-12)
        ewma = max(float(variance), 1e-12)
        for name, v, q in [
            ("rolling_normal", rolling, z * np.sqrt(rolling)),
            ("ewma_normal", ewma, z * np.sqrt(ewma)),
            ("historical", rolling, float(np.quantile(-train, alpha, method="linear"))),
        ]:
            paths[name]["variance"].append(v)
            paths[name]["var"].append(float(q))
        variance = decay * variance + (1 - decay) * x[i] ** 2
    actual = x[start:]
    models = {}
    for name, values in paths.items():
        q, v = np.asarray(values["var"]), np.asarray(values["variance"])
        loss = -actual
        pinball = np.where(loss >= q, alpha * (loss - q), (1 - alpha) * (q - loss))
        models[name] = {
            "coverage": coverage_diagnostics(loss > q, alpha),
            "mean_quantile_loss": float(pinball.mean()),
        }
        if name != "historical":
            models[name]["mean_qlike"] = float(np.mean(np.log(v) + actual**2 / v))
    return {
        "status": "evaluated",
        "protocol": "rolling one-observation-ahead; zero mean; no future observations",
        "window": window,
        "decay": decay,
        "alpha": alpha,
        "start_index": start,
        "models": models,
        "predictions": paths,
        "actual": actual.tolist(),
        "limitations": [
            "Squared one-step return is a noisy variance proxy",
            "Normal tails can underestimate crisis risk",
            "Coverage non-rejection is not proof of correct calibration",
            "No parameter tuning or trading-profit claim",
        ],
    }
